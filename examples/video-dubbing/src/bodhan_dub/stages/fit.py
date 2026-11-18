"""
Duration Fit Controller and Per-Segment Lifecycle Coordinator.
Coordinates TTS synthesis, QA evaluation, retry planning, LLM English condensation,
slack borrowing, and pitch-preserving time stretching (atempo).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.gateway.llm import LLMClient
from bodhan_dub.languages import LanguageRegistry
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.media.audio_proc import time_stretch_audio
from bodhan_dub.policy import guard_llm_output_english_only
from bodhan_dub.stages.qa import QAMetrics, SpeechQA, SynthesisCandidate
from bodhan_dub.stages.tts import CPSCalibrator, TTSSynthesizer
from bodhan_dub.text.spans import verify_numbers_preserved


class CondensedEnglish(BaseModel):
    condensed_en: str = Field(
        default="",
        description="Concise English sentence preserving all entities and numbers.",
    )


class ParaphrasedEnglish(BaseModel):
    paraphrased_en: str = Field(
        default="",
        description="Simplified English paraphrase preserving all entities and numbers.",
    )


class SegmentFitResult(BaseModel):
    segment_id: str
    source_en: str
    final_en: str
    final_translation: str
    raw_duration: float
    final_duration: float
    slot_duration: float
    start_shift: float
    speed_factor: float
    qa_metrics: QAMetrics
    qa_status: str  # passed, degraded, skipped
    fit_status: str  # ok, stretched, borrowed_slack, overrun, skipped
    attempts: int
    reasons: List[str] = Field(default_factory=list)


class SegmentFitController:
    """
    Implements the Per-Segment Duration Fit & QA loop (Diagram 3).
    Ensures speech fits within the available slot and surrounding slacks without
    exceeding 1.2x time stretch, truncating speech, or introducing overlaps.
    """

    def __init__(
        self,
        bodhan_client: Optional[BodhanClient],
        llm_client: Optional[LLMClient],
        logger: PipelineLogger,
        tts_synthesizer: Optional[TTSSynthesizer] = None,
        speech_qa: Optional[SpeechQA] = None,
        calibrator: Optional[CPSCalibrator] = None,
        registry: Optional[LanguageRegistry] = None,
    ):
        self.bodhan_client = bodhan_client
        self.llm_client = llm_client
        self.logger = logger
        self.registry = registry
        self.calibrator = calibrator or CPSCalibrator()
        self.synthesizer = tts_synthesizer or (
            TTSSynthesizer(
                bodhan_client=bodhan_client,
                logger=logger,
                calibrator=self.calibrator,
            )
            if bodhan_client is not None
            else None
        )
        self.speech_qa = speech_qa or (
            SpeechQA(
                bodhan_client=bodhan_client,
                logger=logger,
            )
            if bodhan_client is not None
            else None
        )

    async def _condense_english_source(
        self,
        source_en: str,
        target_char_budget: int,
        lang: str,
        segment_id: str,
    ) -> Optional[str]:
        """
        Uses LLM to condense English source text within a character budget.
        Validates English-only output and number preservation.
        """
        if not self.llm_client:
            return None

        system_prompt = (
            f"Condense the following English sentence into a concise version that can be spoken "
            f"in fewer syllables, aiming for at most {target_char_budget} characters.\n"
            f"Rules:\n"
            f"1. Preserve the core meaning, all numerical values, dates, percentages, and proper names.\n"
            f'2. Return ONLY valid JSON in format: {{"condensed_en": "<condensed English text>"}}\n'
            f"3. Do not include any explanation or non-English text."
        )
        user_prompt = f'Original English: "{source_en}"'
        fallback = CondensedEnglish(condensed_en="")
        try:
            res: CondensedEnglish = await self.llm_client.query_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=CondensedEnglish,
                fallback_value=fallback,
                stage="condense_fit",
                lang=lang,
                segment_id=segment_id,
            )
            condensed = res.condensed_en.strip()
            if not condensed:
                return None

            # Policy Guard: reject any non-English/Indic characters
            guard_llm_output_english_only(condensed)

            # Preserve numbers deterministically
            if not verify_numbers_preserved(source_en, condensed):
                self.logger.warning(
                    "Condensed English omitted numbers; discarding condensation",
                    stage="condense_fit",
                    lang=lang,
                    segment_id=segment_id,
                )
                return None

            return condensed
        except Exception as e:
            self.logger.warning(
                f"LLM condensation failed or was rejected: {e}",
                stage="condense_fit",
                lang=lang,
                segment_id=segment_id,
            )
            return None

    async def _paraphrase_english_source(
        self,
        source_en: str,
        lang: str,
        segment_id: str,
    ) -> Optional[str]:
        """
        Uses LLM to paraphrase English source text for cleaner translation and synthesis.
        """
        if not self.llm_client:
            return None

        system_prompt = (
            "Paraphrase the following English sentence to be simpler, clearer, and easier to pronounce.\n"
            "Rules:\n"
            "1. Keep all numbers, names, and key factual details intact.\n"
            '2. Return ONLY valid JSON in format: {{"paraphrased_en": "<paraphrased English text>"}}\n'
            "3. Do not include any non-English text."
        )
        user_prompt = f'Original English: "{source_en}"'
        fallback = ParaphrasedEnglish(paraphrased_en="")
        try:
            res: ParaphrasedEnglish = await self.llm_client.query_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ParaphrasedEnglish,
                fallback_value=fallback,
                stage="paraphrase_qa",
                lang=lang,
                segment_id=segment_id,
            )
            paraphrased = res.paraphrased_en.strip()
            if not paraphrased:
                return None

            guard_llm_output_english_only(paraphrased)

            if not verify_numbers_preserved(source_en, paraphrased):
                return None

            return paraphrased
        except Exception as e:
            self.logger.warning(
                f"LLM paraphrase failed: {e}",
                stage="paraphrase_qa",
                lang=lang,
                segment_id=segment_id,
            )
            return None

    async def process_segment(
        self,
        segment_id: str,
        source_en: str,
        initial_translation: str,
        slot_duration: float,
        slack_before: float,
        slack_after: float,
        lang: str,
        voice: str,
        code_mixed: bool = False,
    ) -> Tuple[SegmentFitResult, Optional[np.ndarray]]:
        """
        Executes Diagram 3 Per-segment duration fit and QA loop.
        Returns (result_metadata, audio_48k_samples).
        """
        current_en = source_en
        current_tr = initial_translation
        predicted_dur = self.calibrator.predict_duration(current_tr, lang, voice)
        pred_ratio = predicted_dur / max(0.1, slot_duration)

        # 1. Check if pre-synthesis condensation is needed (pred_ratio > 1.3)
        if pred_ratio > 1.3:
            char_budget = max(15, int(len(source_en) * (slot_duration / predicted_dur) * 0.95))
            condensed = await self._condense_english_source(
                source_en, char_budget, lang, segment_id
            )
            if condensed and self.bodhan_client:
                current_en = condensed
                # Re-translate via Indic-Translate
                try:
                    re_tr_text, _ = await self.bodhan_client.translate(
                        text=current_en,
                        target_language_code=lang,
                        source_language_code="en",
                        stage="re_translate_pre_fit",
                        lang=lang,
                        segment_id=segment_id,
                    )
                    if re_tr_text and re_tr_text.strip():
                        current_tr = re_tr_text.strip()
                except Exception as e:
                    self.logger.warning(
                        f"Re-translation failed: {e}", stage="fit", lang=lang, segment_id=segment_id
                    )

        # 2. Synthesis and QA loop (max 3 attempts)
        attempts = 0
        candidates: List[SynthesisCandidate] = []
        chosen_candidate: Optional[SynthesisCandidate] = None

        while attempts < 3:
            attempts += 1
            try:
                if self.synthesizer:
                    audio_48k, actual_dur, p_dur = await self.synthesizer.synthesize(
                        text=current_tr,
                        lang=lang,
                        voice=voice,
                        segment_id=f"{segment_id}_a{attempts}",
                    )
                else:
                    audio_48k = np.array([], dtype=np.float32)
                    actual_dur = 0.0
                    p_dur = predicted_dur
            except Exception as e:
                self.logger.error(
                    f"TTS synthesis exception on attempt {attempts}: {e}",
                    stage="tts",
                    lang=lang,
                    segment_id=segment_id,
                )
                audio_48k = np.array([], dtype=np.float32)
                actual_dur = 0.0
                p_dur = predicted_dur

            # Evaluate QA
            if self.speech_qa:
                qa_metrics = await self.speech_qa.evaluate(
                    audio_48k=audio_48k,
                    reference_text=current_tr,
                    predicted_duration=p_dur,
                    lang=lang,
                    segment_id=f"{segment_id}_a{attempts}",
                )
            else:
                qa_metrics = QAMetrics(
                    cer=0.0,
                    passed=True,
                    has_repetition=False,
                    has_tail_omission=False,
                    duration_ratio=1.0,
                    is_silent=False,
                    is_clipped=False,
                    is_empty=len(audio_48k) == 0,
                )

            cand = SynthesisCandidate(
                attempt=attempts,
                text=current_tr,
                audio_48k=audio_48k,
                duration_s=actual_dur,
                metrics=qa_metrics,
            )
            candidates.append(cand)

            if qa_metrics.passed:
                chosen_candidate = cand
                break

            if attempts < 3:
                # Plan retry action
                action = (
                    self.speech_qa.plan_retry_action(qa_metrics, attempts)
                    if self.speech_qa
                    else "retry_same"
                )
                if action == "paraphrase_source":
                    paraphrased = await self._paraphrase_english_source(source_en, lang, segment_id)
                    if paraphrased and self.bodhan_client:
                        current_en = paraphrased
                        try:
                            tr_text, _ = await self.bodhan_client.translate(
                                text=current_en,
                                target_language_code=lang,
                                source_language_code="en",
                                stage="re_translate_paraphrase",
                                lang=lang,
                                segment_id=segment_id,
                            )
                            if tr_text and tr_text.strip():
                                current_tr = tr_text.strip()
                        except Exception as e:
                            self.logger.warning(
                                f"Re-translation failed: {e}",
                                stage="fit",
                                lang=lang,
                                segment_id=segment_id,
                            )
                elif action == "condense":
                    budget = max(15, int(len(current_en) * 0.8))
                    condensed = await self._condense_english_source(
                        current_en, budget, lang, segment_id
                    )
                    if condensed and self.bodhan_client:
                        current_en = condensed
                        try:
                            tr_text, _ = await self.bodhan_client.translate(
                                text=current_en,
                                target_language_code=lang,
                                source_language_code="en",
                                stage="re_translate_condense",
                                lang=lang,
                                segment_id=segment_id,
                            )
                            if tr_text and tr_text.strip():
                                current_tr = tr_text.strip()
                        except Exception as e:
                            self.logger.warning(
                                f"Re-translation failed: {e}",
                                stage="fit",
                                lang=lang,
                                segment_id=segment_id,
                            )

        # 3. Decision on candidate Usability (Diagram 3: O2)
        qa_status = "passed"
        if chosen_candidate is None:
            # Sort remaining candidates by lowest CER
            valid_cands = [c for c in candidates if len(c.audio_48k) > 0 and not c.metrics.is_empty]
            if valid_cands:
                valid_cands.sort(key=lambda c: c.metrics.cer)
                best = valid_cands[0]
                if best.metrics.cer < 0.45 and not best.metrics.is_silent:
                    chosen_candidate = best
                    qa_status = "degraded"
                else:
                    qa_status = "skipped"
            else:
                qa_status = "skipped"

        # If candidate is unusable / skipped, original audio is kept for this slot
        if chosen_candidate is None or qa_status == "skipped":
            last_metrics = (
                candidates[-1].metrics
                if candidates
                else QAMetrics(
                    cer=1.0, passed=False, is_empty=True, failure_reasons=["all_attempts_failed"]
                )
            )
            result = SegmentFitResult(
                segment_id=segment_id,
                source_en=source_en,
                final_en=current_en,
                final_translation=current_tr,
                raw_duration=0.0,
                final_duration=0.0,
                slot_duration=slot_duration,
                start_shift=0.0,
                speed_factor=1.0,
                qa_metrics=last_metrics,
                qa_status="skipped",
                fit_status="skipped",
                attempts=attempts,
                reasons=["unusable_synthesis_skipped_for_original_audio"],
            )
            return result, None

        # 4. Duration Fit Strategy (Diagram 3: P, Q, R, S, T)
        audio = chosen_candidate.audio_48k
        raw_dur = len(audio) / 48000.0
        fit_ratio = raw_dur / max(0.1, slot_duration)

        if fit_ratio <= 1.0:
            # Natural speed, pad with silence if needed in scheduler
            final_audio = audio
            final_dur = raw_dur
            speed_factor = 1.0
            start_shift = 0.0
            fit_status = "ok"
        elif fit_ratio <= 1.20:
            # Stretch with atempo
            speed_factor = round(fit_ratio, 3)
            final_audio = time_stretch_audio(audio, speed_factor, sample_rate=48000)
            final_dur = len(final_audio) / 48000.0
            start_shift = 0.0
            fit_status = "stretched"
        else:
            # fit_ratio > 1.20: borrow slack, stretch to max 1.20, flag overrun if still over
            borrow_before = min(0.250, max(0.0, slack_before))
            borrow_after = max(0.0, slack_after)
            effective_slot = slot_duration + borrow_before + borrow_after
            eff_ratio = raw_dur / max(0.1, effective_slot)

            if eff_ratio <= 1.20:
                speed_factor = round(max(1.0, eff_ratio), 3)
                final_audio = time_stretch_audio(audio, speed_factor, sample_rate=48000)
                final_dur = len(final_audio) / 48000.0
                excess = max(0.0, final_dur - slot_duration)
                shift = min(borrow_before, excess)
                start_shift = -round(shift, 3)
                fit_status = "borrowed_slack"
            else:
                # Cap speed factor at 1.20 max (Hard Constraint: never speed up beyond 1.20)
                speed_factor = 1.20
                final_audio = time_stretch_audio(audio, speed_factor, sample_rate=48000)
                final_dur = len(final_audio) / 48000.0
                excess = max(0.0, final_dur - slot_duration)
                shift = min(borrow_before, excess)
                start_shift = -round(shift, 3)
                fit_status = "overrun"

        result = SegmentFitResult(
            segment_id=segment_id,
            source_en=source_en,
            final_en=current_en,
            final_translation=chosen_candidate.text,
            raw_duration=round(raw_dur, 3),
            final_duration=round(final_dur, 3),
            slot_duration=round(slot_duration, 3),
            start_shift=start_shift,
            speed_factor=speed_factor,
            qa_metrics=chosen_candidate.metrics,
            qa_status=qa_status,
            fit_status=fit_status,
            attempts=attempts,
            reasons=chosen_candidate.metrics.failure_reasons,
        )

        return result, final_audio
