"""
Master Pipeline Orchestrator for bodhan_dub.
Coordinates shared stages (ingest, VAD, ASR, segmenter, glossary) and forks per-language
pipelines in parallel with full error isolation and resume checkpointing.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from bodhan_dub.config import Settings
from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.gateway.cache import DiskCache
from bodhan_dub.gateway.cost import BudgetExceededError, BudgetGuard, CostLedger
from bodhan_dub.gateway.llm import LLMClient
from bodhan_dub.gateway.resilience import CircuitBreakerOpen
from bodhan_dub.languages import LanguageRegistry
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.manifest import RunManifest
from bodhan_dub.media.chunks import ChunkPlanner
from bodhan_dub.media.extract import extract_audio_for_asr, extract_audio_original
from bodhan_dub.media.mix import mix_dubbed_audio
from bodhan_dub.media.mux import mux_and_verify
from bodhan_dub.media.vad import SileroVADDetector, VADResult
from bodhan_dub.stages.asr import ASRStage
from bodhan_dub.stages.fit import SegmentFitController, SegmentFitResult
from bodhan_dub.stages.glossary import Glossary, GlossaryBuilder
from bodhan_dub.stages.qa import QAMetrics
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
from bodhan_dub.stages.translate import TranslatedSegment, TranslationStage


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
        concurrency: int = 2,
        resume: bool = False,
        separate_music: bool = False,
        speaker_voices: str = "single",
        custom_voices: Optional[Dict[str, str]] = None,
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
        self.concurrency = max(1, concurrency)
        self.resume = resume
        self.separate_music = separate_music
        self.speaker_voices = speaker_voices
        self.custom_voices = custom_voices or {}
        self.separate_music_vocal_path: Optional[Path] = None

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

        # C4: Background music separation if requested
        self.separate_music_vocal_path = None
        if self.separate_music:
            import shutil
            demucs_bin = shutil.which("demucs")
            spleeter_bin = shutil.which("spleeter")
            if demucs_bin or spleeter_bin:
                self.logger.info("Running background music separation (Demucs/Spleeter)...", stage="ingest")
                tool = demucs_bin or spleeter_bin
                try:
                    cmd = [tool, "-n", "htdemucs", "--two-stems", "vocals", "-o", str(self.work_dir), str(audio_orig)]
                    subprocess.run(cmd, capture_output=True, check=True)
                    vocal_candidate = self.work_dir / "htdemucs" / audio_orig.stem / "vocals.wav"
                    if vocal_candidate.exists():
                        self.separate_music_vocal_path = vocal_candidate
                except Exception as e:
                    self.logger.warning(
                        f"Music separation execution failed: {e}. Falling back cleanly to VAD ducking path.",
                        stage="ingest",
                    )
                    self.separate_music_vocal_path = None
            else:
                self.logger.warning(
                    "Music separation requested (--separate-music), but neither demucs nor spleeter is installed or in PATH. "
                    "Falling back cleanly to VAD ducking path.",
                    stage="ingest",
                )

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

        # C6: Speaker-voices auto mode pitch extraction
        if self.speaker_voices == "auto" and english_segments:
            try:
                import soundfile as sf

                from bodhan_dub.stages.segment import estimate_segment_pitch
                audio_16k_data, sr_16k = sf.read(str(audio_16k), dtype="float32")
                for s in english_segments:
                    start_s = max(0, int(s.start * sr_16k))
                    end_s = min(len(audio_16k_data), int(s.end * sr_16k))
                    if end_s > start_s:
                        s.pitch_hz = estimate_segment_pitch(audio_16k_data[start_s:end_s], sr_16k)
            except Exception as e:
                self.logger.warning(f"Pitch estimation failed: {e}", stage="segmenter")

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

    def flush_partial_state(
        self,
        video_path: Path,
        lang_summaries: Optional[Dict[str, LanguageSummary]] = None,
        stage_timings: Optional[Dict[str, float]] = None,
    ) -> None:
        """Flushes manifest, partial report, and logs during interrupt or fatal error."""
        try:
            self.manifest.save(self.out_dir / "manifest.json")
            report = build_pipeline_report(
                run_id=self.manifest.run_id,
                input_file=str(video_path),
                video_duration_s=0.0,
                lang_summaries=lang_summaries or {},
                cost_ledger=self.ledger,
                stage_timings=stage_timings or {},
            )
            report.save(self.out_dir / "report.json")
            self.logger.save_log(self.out_dir / "run.log.jsonl")
        except Exception as e:
            print(f"[ERROR] Failed to flush partial state: {e}", file=sys.stderr)

    async def process_language(
        self,
        lang: str,
        video_path: Path,
        audio_orig_path: Path,
        english_segments: List[SegmentSlot],
        vad_result: VADResult,
        glossary: Optional[Glossary],
        voice: Optional[str] = None,
    ) -> LanguageSummary:
        """
        Executes isolated dubbing pipeline for a single target language:
        Translate -> TTS + Fit + QA -> Schedule -> Mix -> Mux -> Subtitles & Segments JSON.
        """
        import soundfile as sf

        lang_out_dir = self.out_dir / lang
        lang_out_dir.mkdir(parents=True, exist_ok=True)
        if not voice:
            voice = self.custom_voices.get(lang) or self.registry.get_voice(lang)
        self.logger.info(f"Starting dubbing for '{lang}' using voice '{voice}'", stage="pipeline", lang=lang)

        lang_errors: List[str] = []

        # 1. Translation Stage
        trans_stage = TranslationStage(
            bodhan_client=self.bodhan_client,
            llm_client=self.llm_client,
            registry=self.registry,
            logger=self.logger,
            code_mixed=self.code_mixed,
        )

        try:
            translations = await trans_stage.run(
                segments=english_segments,
                target_lang=lang,
                glossary=glossary or Glossary(),
            )
        except (BudgetExceededError, CircuitBreakerOpen) as e:
            self.logger.error(f"Translation stopped for '{lang}': {e}", stage="translate", lang=lang)
            lang_errors.append(str(e))
            translations = [
                TranslatedSegment(
                    id=seg.id,
                    index=seg.index,
                    source_text=seg.text,
                    target_text="",
                    target_lang=lang,
                    status="translation_failed",
                    flag_reason=str(e),
                )
                for seg in english_segments
            ]

        # 2. TTS, QA, and Duration Fit Controller
        fit_controller = SegmentFitController(
            bodhan_client=self.bodhan_client,
            llm_client=self.llm_client,
            logger=self.logger,
            registry=self.registry,
        )

        lang_work_dir = self.work_dir / lang
        lang_work_dir.mkdir(parents=True, exist_ok=True)

        # C6: Speaker-voices auto mode per-segment voice mapping
        segment_voices = [voice] * len(english_segments)
        if self.speaker_voices == "auto" and english_segments:
            from bodhan_dub.stages.segment import cluster_speaker_voices
            pitches = [s.pitch_hz if s.pitch_hz is not None else 160.0 for s in english_segments]
            segment_voices = cluster_speaker_voices(pitches, lang, self.registry)

        async def _fit_single_segment(idx: int, seg: SegmentSlot, tr_info: TranslatedSegment, seg_voice: str):
            if tr_info.status == "translation_failed" or not tr_info.target_text.strip():
                self.logger.warning(
                    f"Translation failed or empty for segment {seg.segment_id}; skipping TTS to keep original audio",
                    stage="fit",
                    lang=lang,
                    segment_id=seg.segment_id,
                )
                res = SegmentFitResult(
                    segment_id=seg.segment_id,
                    source_en=seg.text,
                    final_en=seg.text,
                    final_translation="",
                    raw_duration=0.0,
                    final_duration=0.0,
                    slot_duration=seg.slot_duration,
                    start_shift=0.0,
                    speed_factor=1.0,
                    qa_metrics=QAMetrics(
                        cer=1.0,
                        passed=False,
                        is_empty=True,
                        failure_reasons=[tr_info.flag_reason or "translation_failed"],
                    ),
                    qa_status="skipped",
                    fit_status="skipped",
                    attempts=0,
                    reasons=[tr_info.flag_reason or "translation_failed"],
                )
                return idx, res, None

            # C4: Check checkpoint for --resume
            cached_wav = lang_work_dir / f"{seg.segment_id}.wav"
            if self.resume and self.manifest.is_segment_completed(lang, seg.segment_id) and cached_wav.exists():
                try:
                    c_audio, c_sr = sf.read(str(cached_wav), dtype="float32")
                    c_dur = len(c_audio) / float(c_sr)
                    self.logger.info(
                        f"Resuming segment {seg.segment_id} from checkpoint ({c_dur:.2f}s)",
                        stage="fit",
                        lang=lang,
                        segment_id=seg.segment_id,
                    )
                    res = SegmentFitResult(
                        segment_id=seg.segment_id,
                        source_en=seg.text,
                        final_en=seg.text,
                        final_translation=tr_info.target_text,
                        raw_duration=c_dur,
                        final_duration=c_dur,
                        slot_duration=seg.slot_duration,
                        start_shift=0.0,
                        speed_factor=1.0,
                        qa_metrics=QAMetrics(cer=0.0, passed=True, failure_reasons=[]),
                        qa_status="passed",
                        fit_status="fit_exact",
                        attempts=1,
                        reasons=[],
                    )
                    return idx, res, c_audio
                except Exception:
                    pass

            terms_to_protect = (
                glossary.get_terms_for_text(seg.text) if glossary else []
            )
            res, audio_48k = await fit_controller.process_segment(
                segment_id=seg.segment_id,
                source_en=seg.text,
                initial_translation=tr_info.target_text,
                slot_duration=seg.slot_duration,
                slack_before=seg.slack_before,
                slack_after=seg.slack_after,
                lang=lang,
                voice=seg_voice,
                code_mixed=self.code_mixed,
                terms_to_protect=terms_to_protect,
            )
            if audio_48k is not None:
                try:
                    sf.write(str(cached_wav), audio_48k, 48000)
                except Exception:
                    pass
            self.manifest.record_completed_segment(lang, seg.segment_id)
            return idx, res, audio_48k

        # C3: Parallel segment synthesis bounded by settings.semaphore_tts in bodhan_client
        fit_tasks = [
            _fit_single_segment(i, seg, tr_info, segment_voices[i])
            for i, (seg, tr_info) in enumerate(zip(english_segments, translations))
        ]
        fit_raw_results = await asyncio.gather(*fit_tasks, return_exceptions=True)

        fit_results = []
        audio_buffers = []
        retries_count = 0

        for idx, item in enumerate(fit_raw_results):
            seg = english_segments[idx]
            tr_info = translations[idx]
            if isinstance(item, Exception):
                self.logger.error(
                    f"Processing stopped on segment {seg.segment_id} for '{lang}': {item}",
                    stage="fit",
                    lang=lang,
                    segment_id=seg.segment_id,
                )
                lang_errors.append(str(item))
                res = SegmentFitResult(
                    segment_id=seg.segment_id,
                    source_en=seg.text,
                    final_en=seg.text,
                    final_translation=tr_info.target_text,
                    raw_duration=0.0,
                    final_duration=0.0,
                    slot_duration=seg.slot_duration,
                    start_shift=0.0,
                    speed_factor=1.0,
                    qa_metrics=QAMetrics(
                        cer=1.0,
                        passed=False,
                        is_empty=True,
                        failure_reasons=[str(item)],
                    ),
                    qa_status="skipped",
                    fit_status="skipped",
                    attempts=0,
                    reasons=[str(item)],
                )
                fit_results.append(res)
                audio_buffers.append(None)
            else:
                _i, res, audio_48k = item
                fit_results.append(res)
                audio_buffers.append(audio_48k)
                retries_count += max(0, res.attempts - 1)

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
            separate_music_vocal_path=self.separate_music_vocal_path,
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

        flagged_count = sum(
            1 for idx, r in enumerate(fit_results)
            if (idx < len(translations) and translations[idx].status == "translation_flagged")
            or "flagged" in r.reasons
            or r.qa_status == "degraded"
        )
        overrun_count = sum(1 for r in fit_results if r.fit_status == "overrun")

        has_failed_video = not mux_res.verified or not target_mp4_path.exists()
        all_speech_skipped = (len(english_segments) > 0 and sched_res.placed_segments == 0)

        if has_failed_video or all_speech_skipped:
            status = "failed"
        elif (
            sched_res.degraded_segments > 0
            or sched_res.skipped_segments > 0
            or flagged_count > 0
            or overrun_count > 0
            or any(t.status in ("translation_flagged", "translation_failed") for t in translations)
        ):
            status = "degraded"
        else:
            status = "ok"

        summary = LanguageSummary(
            status=status,
            total_segments=sched_res.total_segments,
            placed_segments=sched_res.placed_segments,
            degraded_segments=sched_res.degraded_segments,
            skipped_segments=sched_res.skipped_segments,
            flagged_segments=flagged_count,
            overrun_segments=overrun_count,
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
            errors=lang_errors,
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
        custom_voices: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Executes end-to-end pipeline.
        Returns process exit code: 0 = ok/degraded, 1 = at least one language failed.
        """
        start_time = time.time()
        stage_timings: Dict[str, float] = {}

        if custom_voices:
            self.custom_voices.update(custom_voices)

        if max_cost_inr is not None:
            self.budget_guard.set_budget(max_cost_inr)

        video_path = Path(video_path).resolve()
        self.logger.info(f"Starting dubbing pipeline for {video_path.name} into {target_langs}", stage="pipeline")

        # 1. Run Shared Stages
        t0 = time.time()
        vad_res, english_segments, audio_orig_path, glossary = await self.run_shared_stages(video_path, target_langs)
        stage_timings["shared_stages"] = time.time() - t0

        # Handle silent / music-only video (ok_no_speech)
        if not vad_res.has_speech:
            self.logger.info("Video contains no detected speech; generating passthrough outputs with AI-dubbed metadata", stage="pipeline")
            lang_summaries = {}
            for lang in target_langs:
                lang_dir = self.out_dir / lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                target_video = lang_dir / f"video.{lang}.mp4"

                # Re-mux with ffmpeg -c copy -metadata comment="AI-dubbed with Bodhan Indic-Speak"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-c", "copy",
                    "-metadata", "comment=AI-dubbed with Bodhan Indic-Speak",
                    "-metadata", "title=AI-dubbed with Bodhan Indic-Speak",
                    "-movflags", "+faststart",
                    str(target_video),
                ]
                res = subprocess.run(cmd, capture_output=True)
                if res.returncode != 0 or not target_video.exists():
                    shutil.copy2(video_path, target_video)

                write_target_segments_json([], lang_dir / f"segments.{lang}.json")
                write_target_subtitles([], lang_dir / f"subtitles.{lang}.srt", lang_dir / f"subtitles.{lang}.vtt")
                lang_summaries[lang] = LanguageSummary(
                    status="ok_no_speech",
                    total_segments=0,
                    placed_segments=0,
                    degraded_segments=0,
                    skipped_segments=0,
                    flagged_segments=0,
                    overrun_segments=0,
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

        # Handle case where VAD found speech but zero segments survived ASR filtering
        if len(english_segments) == 0:
            self.logger.warning(
                "VAD detected speech regions, but zero segments survived ASR filtering and hallucination checks!",
                stage="pipeline",
            )
            lang_summaries = {}
            for lang in target_langs:
                lang_dir = self.out_dir / lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                target_video = lang_dir / f"video.{lang}.mp4"

                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-c", "copy",
                    "-metadata", "comment=AI-dubbed with Bodhan Indic-Speak",
                    "-metadata", "title=AI-dubbed with Bodhan Indic-Speak",
                    "-movflags", "+faststart",
                    str(target_video),
                ]
                res = subprocess.run(cmd, capture_output=True)
                if res.returncode != 0 or not target_video.exists():
                    shutil.copy2(video_path, target_video)

                write_target_segments_json([], lang_dir / f"segments.{lang}.json")
                write_target_subtitles([], lang_dir / f"subtitles.{lang}.srt", lang_dir / f"subtitles.{lang}.vtt")
                lang_summaries[lang] = LanguageSummary(
                    status="degraded",
                    total_segments=0,
                    placed_segments=0,
                    degraded_segments=0,
                    skipped_segments=0,
                    flagged_segments=0,
                    overrun_segments=0,
                    integrated_lufs=-70.0,
                    duration_delta_ms=0.0,
                    retries_count=0,
                    errors=["VAD found speech but zero segments survived ASR filtering"],
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

        # 2. Per-Language Parallel Fork with Concurrency Semaphore (C3)
        t_lang_start = time.time()
        lang_sem = asyncio.Semaphore(self.concurrency)

        async def _run_lang_worker(lang_code: str):
            async with lang_sem:
                v = (custom_voices.get(lang_code) if custom_voices else None) or self.custom_voices.get(lang_code)
                return await self.process_language(
                    lang=lang_code,
                    video_path=video_path,
                    audio_orig_path=audio_orig_path,
                    english_segments=english_segments,
                    vad_result=vad_res,
                    glossary=glossary,
                    voice=v,
                )

        tasks = [asyncio.create_task(_run_lang_worker(lang_code)) for lang_code in target_langs]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except (asyncio.CancelledError, KeyboardInterrupt):
            for t in tasks:
                if not t.done():
                    t.cancel()
            self.flush_partial_state(video_path, stage_timings=stage_timings)
            raise

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
                    flagged_segments=0,
                    overrun_segments=0,
                    integrated_lufs=-70.0,
                    duration_delta_ms=0.0,
                    retries_count=0,
                    errors=[str(res)],
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
