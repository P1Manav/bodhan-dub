"""
Master Pipeline Orchestrator for bodhan_dub.
Coordinates shared stages (ingest, VAD, ASR, segmenter, glossary) and forks per-language
pipelines in parallel with full error isolation and resume checkpointing.
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from bodhan_dub.config import Settings
from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.gateway.cache import DiskCache
from bodhan_dub.gateway.cost import BudgetGuard, CostLedger
from bodhan_dub.gateway.llm import LLMClient
from bodhan_dub.languages import LanguageRegistry
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.manifest import RunManifest
from bodhan_dub.media.chunks import ChunkPlanner
from bodhan_dub.media.extract import extract_audio_for_asr, extract_audio_original
from bodhan_dub.media.mix import mix_dubbed_audio
from bodhan_dub.media.mux import mux_and_verify
from bodhan_dub.media.vad import SileroVADDetector, VADResult
from bodhan_dub.stages.asr import ASRStage
from bodhan_dub.stages.fit import SegmentFitController
from bodhan_dub.stages.glossary import Glossary, GlossaryBuilder
from bodhan_dub.stages.report import LanguageSummary, build_pipeline_report
from bodhan_dub.stages.schedule import TimelineScheduler
from bodhan_dub.stages.segment import Segmenter, SegmentSlot
from bodhan_dub.stages.subtitles import (
    write_srt,
    write_target_segments_json,
    write_target_subtitles,
    write_transcript_en_json,
    write_vtt,
)
from bodhan_dub.stages.translate import TranslationStage


class PipelineOrchestrator:
    """Manages DAG execution, shared stages, and per-language isolated pipelines."""

    def __init__(
        self,
        settings: Settings,
        registry: LanguageRegistry,
        logger: PipelineLogger,
        manifest: RunManifest,
        out_dir: Path,
        work_dir: Optional[Path] = None,
        no_cache: bool = False,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        glossary_path: Optional[Path] = None,
        code_mixed: bool = False,
        keep_work: bool = False,
    ):
        self.settings = settings
        self.registry = registry
        self.logger = logger
        self.manifest = manifest
        self.out_dir = Path(out_dir)
        self.work_dir = Path(work_dir) if work_dir else self.out_dir / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.glossary_path = glossary_path
        self.code_mixed = code_mixed
        self.keep_work = keep_work

        self.cache = DiskCache(cache_dir=settings.cache_dir, enabled=not no_cache)
        self.ledger = CostLedger(
            cost_asr_per_min=settings.cost_asr_per_minute,
            cost_translate_per_10k=settings.cost_translate_per_10k_tokens,
            cost_tts_per_10k=settings.cost_tts_per_10k_chars,
        )
        self.budget_guard = BudgetGuard(self.ledger)

        self.bodhan_client = BodhanClient(
            settings=settings,
            cache=self.cache,
            ledger=self.ledger,
            budget_guard=self.budget_guard,
            logger=logger,
            transport=transport,
        )

        self.llm_client = LLMClient(
            settings=settings,
            cache=self.cache,
            ledger=self.ledger,
            logger=logger,
            transport=transport,
        )

    async def close(self) -> None:
        await self.bodhan_client.close()
        await self.llm_client.close()

    async def run_shared_stages(
        self,
        video_path: Path,
        target_langs: List[str],
    ) -> Tuple[VADResult, List[SegmentSlot], Path, Optional[Glossary]]:
        """
        Executes shared stages: Ingest -> VAD -> Chunking -> ASR -> Segmentation -> Glossary.
        Returns (vad_result, english_segments, audio_original_path, glossary).
        """
        # 1. Ingest: Audio extraction
        self.manifest.mark_stage_start("ingest")
        self.logger.info("Extracting 16kHz audio for ASR and 48kHz audio for background mix...", stage="ingest")
        audio_16k = self.work_dir / "audio_16k.wav"
        audio_orig = self.work_dir / "audio_original.wav"

        extract_audio_for_asr(video_path, audio_16k)
        extract_audio_original(video_path, audio_orig)
        self.manifest.mark_stage_completed("ingest")

        # 2. VAD: Speech boundary detection
        self.manifest.mark_stage_start("vad")
        self.logger.info("Running Voice Activity Detection (Silero/ffmpeg)...", stage="vad")
        vad_detector = SileroVADDetector()
        vad_result = vad_detector.process(audio_16k)
        self.logger.info(
            f"VAD complete ({vad_result.engine_used}): {len(vad_result.speech_regions)} speech regions, "
            f"total speech {vad_result.total_speech_duration:.1f}s / {vad_result.total_duration:.1f}s",
            stage="vad",
        )
        self.manifest.mark_stage_completed("vad")

        # Handle silent / music-only video
        if not vad_result.has_speech:
            self.logger.warning("No speech detected in video! Flagging ok_no_speech.", stage="vad")
            write_transcript_en_json([], self.out_dir / "transcript.en.json")
            return vad_result, [], audio_orig, None

        # 3. Chunk Planner
        self.manifest.mark_stage_start("chunk_planner")
        planner = ChunkPlanner(target_duration=25.0, hard_max_duration=29.0)
        chunks = planner.plan_chunks(vad_result, audio_path=audio_16k)
        self.logger.info(f"Planned {len(chunks)} bounded ASR chunks (hard max 29s)", stage="chunk_planner")
        self.manifest.mark_stage_completed("chunk_planner")

        # 4. ASR Stage
        self.manifest.mark_stage_start("asr")
        self.logger.info(f"Transcribing {len(chunks)} chunks with Indic-Transcribe...", stage="asr")
        asr_stage = ASRStage(self.bodhan_client, self.logger)
        raw_segments = await asr_stage.run(audio_16k, chunks, vad_result)
        self.logger.info(f"ASR complete: produced {len(raw_segments)} raw segments", stage="asr")
        self.manifest.mark_stage_completed("asr")

        # 5. Segmenter with Slot & Slack
        self.manifest.mark_stage_start("segmenter")
        self.logger.info("Running sentence segmentation and computing slot/slack...", stage="segmenter")
        segmenter = Segmenter(self.llm_client, self.logger)
        english_segments = await segmenter.run(raw_segments, total_audio_duration=vad_result.total_duration)
        self.logger.info(f"Segmentation complete: finalized {len(english_segments)} sentence slots", stage="segmenter")
        self.manifest.mark_stage_completed("segmenter")

        # 6. Write English Outputs
        write_transcript_en_json(english_segments, self.out_dir / "transcript.en.json")
        write_srt(english_segments, self.out_dir / "subtitles.en.srt")
        write_vtt(english_segments, self.out_dir / "subtitles.en.vtt")
        self.logger.info(f"Wrote English transcript and subtitles to {self.out_dir}", stage="segmenter")

        # 7. Glossary Builder
        self.manifest.mark_stage_start("glossary")
        glossary_builder = GlossaryBuilder(
            llm_client=self.llm_client,
            bodhan_client=self.bodhan_client,
            logger=self.logger,
            user_glossary_path=self.glossary_path,
        )
        glossary = await glossary_builder.build(english_segments, target_langs)
        self.manifest.mark_stage_completed("glossary")

        return vad_result, english_segments, audio_orig, glossary

    async def process_language(
        self,
        lang: str,
        video_path: Path,
        audio_orig_path: Path,
        english_segments: List[SegmentSlot],
        vad_result: VADResult,
        glossary: Optional[Glossary],
    ) -> LanguageSummary:
        """
        Executes isolated dubbing pipeline for a single target language:
        Translate -> TTS + Fit + QA -> Schedule -> Mix -> Mux -> Subtitles & Segments JSON.
        """
        lang_out_dir = self.out_dir / lang
        lang_out_dir.mkdir(parents=True, exist_ok=True)
        voice = self.registry.get_voice(lang)
        self.logger.info(f"Starting dubbing for '{lang}' using voice '{voice}'", stage="pipeline", lang=lang)

        # 1. Translation Stage
        trans_stage = TranslationStage(
            bodhan_client=self.bodhan_client,
            llm_client=self.llm_client,
            registry=self.registry,
            logger=self.logger,
            code_mixed=self.code_mixed,
        )
        translations = await trans_stage.run(
            segments=english_segments,
            target_lang=lang,
            glossary=glossary or Glossary(),
        )

        # 2. TTS, QA, and Duration Fit Controller
        fit_controller = SegmentFitController(
            bodhan_client=self.bodhan_client,
            llm_client=self.llm_client,
            logger=self.logger,
            registry=self.registry,
        )

        fit_results = []
        audio_buffers = []
        retries_count = 0

        for seg, tr_info in zip(english_segments, translations):
            res, audio_48k = await fit_controller.process_segment(
                segment_id=seg.segment_id,
                source_en=seg.text,
                initial_translation=tr_info.target_text,
                slot_duration=seg.slot_duration,
                slack_before=seg.slack_before,
                slack_after=seg.slack_after,
                lang=lang,
                voice=voice,
                code_mixed=self.code_mixed,
            )
            fit_results.append(res)
            audio_buffers.append(audio_48k)
            retries_count += (res.attempts - 1)
            self.manifest.record_completed_segment(lang, seg.segment_id)

        # 3. Timeline Scheduler
        scheduler = TimelineScheduler(sample_rate=48000, max_sync_drift_s=0.300)
        sched_res, speech_stem_48k = scheduler.schedule(
            lang=lang,
            source_segments=english_segments,
            fit_results=fit_results,
            audio_buffers=audio_buffers,
            total_duration_s=vad_result.total_duration,
            logger=self.logger,
        )

        # 4. Audio Mixer
        speech_intervals_to_mute: List[Tuple[float, float]] = []
        for seg, sched_seg in zip(english_segments, sched_res.segments):
            if not sched_seg.is_skipped:
                speech_intervals_to_mute.append((seg.start, seg.end))

        dubbed_wav_path = self.work_dir / f"dubbed_{lang}.wav"
        integrated_lufs = mix_dubbed_audio(
            original_audio_path=audio_orig_path,
            speech_stem_48k=speech_stem_48k,
            speech_intervals_to_mute=speech_intervals_to_mute,
            output_wav_path=dubbed_wav_path,
            target_lufs=-16.0,
            sample_rate=48000,
        )

        # 5. Video Muxer
        target_mp4_path = lang_out_dir / f"video.{lang}.mp4"
        mux_res = mux_and_verify(
            video_path=video_path,
            audio_path=dubbed_wav_path,
            output_mp4_path=target_mp4_path,
            lang=lang,
            logger=self.logger,
        )

        # 6. Subtitles and Segments Output
        segments_json_data = []
        for seg, fit_res, sched_seg in zip(english_segments, fit_results, sched_res.segments):
            segments_json_data.append({
                "segment_id": seg.segment_id,
                "source_text": seg.text,
                "translated_text": fit_res.final_translation,
                "start": sched_seg.placed_start,
                "end": sched_seg.placed_end,
                "audio_duration": fit_res.final_duration,
                "speed_factor": fit_res.speed_factor,
                "start_shift": fit_res.start_shift,
                "qa_status": fit_res.qa_status,
                "fit_status": fit_res.fit_status,
                "cer": fit_res.qa_metrics.cer,
                "qa_reasons": fit_res.reasons,
            })

        write_target_segments_json(segments_json_data, lang_out_dir / f"segments.{lang}.json")
        write_target_subtitles(
            segments_json_data,
            srt_path=lang_out_dir / f"subtitles.{lang}.srt",
            vtt_path=lang_out_dir / f"subtitles.{lang}.vtt",
        )

        # Compute summary statistics
        stretches = [s.speed_factor for s in sched_res.segments if not s.is_skipped]
        cers = [s.qa_metrics.cer for s in fit_results if not s.qa_metrics.is_empty]

        status = "ok"
        if sched_res.degraded_segments > 0:
            status = "degraded"
        if not mux_res.verified:
            status = "failed"

        summary = LanguageSummary(
            status=status,
            total_segments=sched_res.total_segments,
            placed_segments=sched_res.placed_segments,
            degraded_segments=sched_res.degraded_segments,
            skipped_segments=sched_res.skipped_segments,
            stretch_stats={
                "min": float(np.min(stretches)) if stretches else 1.0,
                "max": float(np.max(stretches)) if stretches else 1.0,
                "mean": float(np.mean(stretches)) if stretches else 1.0,
            },
            sync_offset_stats={
                "max_ms": sched_res.max_sync_offset_ms,
                "p95_ms": sched_res.p95_sync_offset_ms,
            },
            cer_stats={
                "mean": float(np.mean(cers)) if cers else 0.0,
                "p95": float(np.percentile(cers, 95)) if cers else 0.0,
            },
            integrated_lufs=round(integrated_lufs, 2),
            duration_delta_ms=round(mux_res.duration_delta_ms, 1),
            retries_count=retries_count,
        )

        self.logger.info(
            f"Finished dubbing for '{lang}': status={summary.status}, placed={summary.placed_segments}, "
            f"max_sync={summary.sync_offset_stats.get('max_ms')}ms, LUFS={summary.integrated_lufs}",
            stage="pipeline",
            lang=lang,
        )
        return summary

    async def run(
        self,
        video_path: Path,
        target_langs: List[str],
        max_cost_inr: Optional[float] = None,
    ) -> int:
        """
        Executes end-to-end pipeline.
        Returns process exit code: 0 = ok/degraded, 1 = at least one language failed.
        """
        start_time = time.time()
        stage_timings: Dict[str, float] = {}

        if max_cost_inr is not None:
            self.budget_guard.set_budget(max_cost_inr)

        video_path = Path(video_path).resolve()
        self.logger.info(f"Starting dubbing pipeline for {video_path.name} into {target_langs}", stage="pipeline")

        # 1. Run Shared Stages
        t0 = time.time()
        vad_res, english_segments, audio_orig_path, glossary = await self.run_shared_stages(video_path, target_langs)
        stage_timings["shared_stages"] = time.time() - t0

        # Handle silent video (ok_no_speech)
        if not vad_res.has_speech:
            self.logger.info("Video contains no detected speech; generating passthrough outputs", stage="pipeline")
            lang_summaries = {}
            for lang in target_langs:
                lang_dir = self.out_dir / lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, lang_dir / f"video.{lang}.mp4")
                write_target_segments_json([], lang_dir / f"segments.{lang}.json")
                write_target_subtitles([], lang_dir / f"subtitles.{lang}.srt", lang_dir / f"subtitles.{lang}.vtt")
                lang_summaries[lang] = LanguageSummary(
                    status="ok",
                    total_segments=0,
                    placed_segments=0,
                    degraded_segments=0,
                    skipped_segments=0,
                    integrated_lufs=-70.0,
                    duration_delta_ms=0.0,
                    retries_count=0,
                )

            report = build_pipeline_report(
                run_id=self.manifest.run_id,
                input_file=str(video_path),
                video_duration_s=vad_res.total_duration,
                lang_summaries=lang_summaries,
                cost_ledger=self.ledger,
                stage_timings=stage_timings,
            )
            report.save(self.out_dir / "report.json")
            self.manifest.save(self.out_dir / "manifest.json")
            self.logger.save_log(self.out_dir / "run.log.jsonl")
            return 0

        # 2. Per-Language Parallel Fork
        t_lang_start = time.time()
        tasks = []
        for lang in target_langs:
            tasks.append(
                self.process_language(
                    lang=lang,
                    video_path=video_path,
                    audio_orig_path=audio_orig_path,
                    english_segments=english_segments,
                    vad_result=vad_res,
                    glossary=glossary,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        stage_timings["per_language_stages"] = time.time() - t_lang_start

        lang_summaries: Dict[str, LanguageSummary] = {}
        any_failed = False

        for lang, res in zip(target_langs, results):
            if isinstance(res, Exception):
                self.logger.error(f"Language pipeline '{lang}' failed with exception: {res}", stage="pipeline", lang=lang)
                any_failed = True
                lang_summaries[lang] = LanguageSummary(
                    status="failed",
                    total_segments=len(english_segments),
                    placed_segments=0,
                    degraded_segments=0,
                    skipped_segments=len(english_segments),
                    integrated_lufs=-70.0,
                    duration_delta_ms=0.0,
                    retries_count=0,
                )
            else:
                lang_summaries[lang] = res
                if res.status == "failed":
                    any_failed = True

        stage_timings["total_pipeline"] = time.time() - start_time

        # 3. Generate Final Report and Save Logs
        report = build_pipeline_report(
            run_id=self.manifest.run_id,
            input_file=str(video_path),
            video_duration_s=vad_res.total_duration,
            lang_summaries=lang_summaries,
            cost_ledger=self.ledger,
            stage_timings=stage_timings,
        )
        report.save(self.out_dir / "report.json")
        self.manifest.save(self.out_dir / "manifest.json")
        self.logger.save_log(self.out_dir / "run.log.jsonl")

        # Clean work directory if requested
        if not self.keep_work and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)

        self.logger.info(
            f"Pipeline run {self.manifest.run_id} completed in {stage_timings['total_pipeline']:.1f}s. "
            f"Overall status: {report.overall_status}. Wrote report to {self.out_dir / 'report.json'}",
            stage="pipeline",
        )

        return 1 if any_failed else 0
