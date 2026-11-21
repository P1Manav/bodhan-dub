"""
Translation stage for bodhan_dub using Indic-Translate.
Implements structural validation checks, placeholder management, 4-step retry ladder,
and English-only LLM back-translation judging.
"""

from __future__ import annotations

import asyncio
import unicodedata
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from bodhan_dub.gateway.bodhan import BodhanClient
from bodhan_dub.gateway.cost import BudgetExceededError
from bodhan_dub.gateway.llm import LLMClient
from bodhan_dub.gateway.resilience import CircuitBreakerOpen
from bodhan_dub.languages import LanguageRegistry
from bodhan_dub.logging import PipelineLogger
from bodhan_dub.policy import guard_llm_output_english_only
from bodhan_dub.stages.glossary import Glossary
from bodhan_dub.stages.segment import SegmentSlot
from bodhan_dub.text.normalize import detect_repeated_ngrams
from bodhan_dub.text.spans import protect_terms, restore_terms, verify_numbers_preserved


class TranslatedSegment(BaseModel):
    id: str
    index: int
    source_text: str
    target_text: str
    target_lang: str
    status: str = "ok"  # "ok", "translation_flagged", "translation_failed"
    flag_reason: Optional[str] = None
    back_translation: Optional[str] = None
    judge_verdict: Optional[str] = "ok"
    judge_evidence: Optional[str] = None
    attempt: int = 1


class ParaphrasedEnglish(BaseModel):
    paraphrase: str = Field(
        description="Simplified, clear English paraphrase preserving all entities and numbers."
    )


class BackTranslationVerdict(BaseModel):
    verdict: str = Field(description="Verdict: 'ok', 'mistranslation', 'omission', or 'addition'")
    evidence: str = Field(description="Explanation of the discrepancy if not ok")


def check_dominant_script(
    text: str,
    target_lang: str,
    registry: LanguageRegistry,
    min_indic_ratio: float = 0.5,
) -> Tuple[bool, str]:
    """
    Checks that the text is written in the expected Indic script.
    Allows Latin letters (e.g. acronyms or brand names) but ensures target script dominates
    and no other Indic script dominates.
    """
    info = registry.get_info(target_lang)
    if not info or not info.unicode_range:
        return True, "ok"

    range_start, range_end = info.unicode_range
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return False, "no_letters"

    def is_in_target_range(code_pt: int) -> bool:
        if target_lang == "ur" or (info.script and info.script in ("Nastaliq", "Arabic")):
            return (0x0600 <= code_pt <= 0x06FF) or (0x08A0 <= code_pt <= 0x08FF)
        return range_start <= code_pt <= range_end

    indic_count = sum(1 for ch in letters if is_in_target_range(ord(ch)))
    latin_count = sum(1 for ch in letters if ord(ch) < 0x024F)

    # Check if letters from a DIFFERENT Indic block dominate
    for r in registry.indic_script_ranges:
        r_s, r_e = r[0], r[1]
        is_target_block = False
        if target_lang == "ur" and r_s in (0x0600, 0x08A0):
            is_target_block = True
        elif r_s == range_start and r_e == range_end:
            is_target_block = True

        if not is_target_block:
            other_count = sum(1 for ch in letters if r_s <= ord(ch) <= r_e)
            if other_count > indic_count and other_count > 0:
                return False, f"dominant_script_mismatch (different_script_dominated: {other_count} vs {indic_count})"

    if indic_count == 0:
        if latin_count > 0:
            return False, "dominant_script_mismatch (latin_only)"
        return False, "dominant_script_mismatch (no_target_script)"

    # Discount allowed Latin terms
    non_latin_count = max(1, len(letters) - latin_count)
    indic_ratio = indic_count / float(non_latin_count)
    if indic_ratio < min_indic_ratio:
        return False, f"dominant_script_mismatch (indic_ratio={indic_ratio:.2f} < {min_indic_ratio})"

    return True, "ok"


def run_translation_structural_checks(
    source_text: str,
    target_text: str,
    target_lang: str,
    registry: LanguageRegistry,
) -> Tuple[bool, Optional[str]]:
    """
    Runs deterministic validation checks on translated Indic text:
    1. Non-empty
    2. Dominant script
    3. Numbers preserved
    4. Length ratio (0.4 to 3.0)
    5. No repeated n-grams / looping
    6. No leftover placeholders
    7. Not verbatim copy of source
    """
    t_clean = target_text.strip()
    s_clean = source_text.strip()

    if not t_clean:
        return False, "empty_translation"

    if t_clean.lower() == s_clean.lower():
        return False, "verbatim_copy_of_source"

    if "__TERM_" in t_clean:
        return False, "leftover_placeholders"

    # Length ratio check
    ratio = len(t_clean) / max(1.0, float(len(s_clean)))
    if ratio < 0.4 or ratio > 3.2:
        return False, f"abnormal_length_ratio_{ratio:.2f}"

    # Dominant script check
    script_ok, reason = check_dominant_script(t_clean, target_lang, registry)
    if not script_ok:
        return False, reason

    # Numbers preservation
    if not verify_numbers_preserved(s_clean, t_clean):
        return False, "number_mismatch"

    # Repeated n-gram looping
    is_loop, phrase = detect_repeated_ngrams(t_clean, min_n=1, max_n=3, repeat_threshold=3)
    if is_loop:
        return False, f"repeated_ngram_loop_{phrase}"

    return True, None


class TranslationStage:
    """Manages translation of segments, checks, retry ladder, and judge."""

    def __init__(
        self,
        bodhan_client: BodhanClient,
        llm_client: LLMClient,
        registry: LanguageRegistry,
        logger: PipelineLogger,
        code_mixed: bool = False,
    ):
        self.bodhan_client = bodhan_client
        self.llm_client = llm_client
        self.registry = registry
        self.logger = logger
        self.code_mixed = code_mixed

    async def paraphrase_source_english(
        self,
        english_text: str,
        lang: str,
        segment_id: str,
        terms_to_protect: Optional[List[str]] = None,
    ) -> str:
        """Uses LLM to simplify or paraphrase English text when translation fails."""
        if not self.llm_client.settings.llm_api_key:
            return english_text

        try:
            paraphrase = await self.llm_client.paraphrase_english(
                source_en=english_text,
                lang=lang,
                segment_id=segment_id,
            )
            if not paraphrase:
                return english_text

            # Verify English-only
            guard_llm_output_english_only(paraphrase)

            # Verify numbers preserved
            if not verify_numbers_preserved(english_text, paraphrase):
                self.logger.warning(
                    "Paraphrase omitted numbers; falling back to original",
                    stage="translate",
                    lang=lang,
                    segment_id=segment_id,
                )
                return english_text

            # Verify terms preserved
            if terms_to_protect:
                for term in terms_to_protect:
                    if term.lower() not in paraphrase.lower():
                        self.logger.warning(
                            f"Paraphrase omitted glossary term '{term}'; falling back to original",
                            stage="translate",
                            lang=lang,
                            segment_id=segment_id,
                        )
                        return english_text

            return paraphrase
        except (BudgetExceededError, CircuitBreakerOpen):
            raise
        except Exception as e:
            self.logger.warning(
                f"Paraphrase failed: {e}; falling back to original",
                stage="translate",
                lang=lang,
                segment_id=segment_id,
            )
            return english_text

    async def judge_back_translation(
        self,
        source_en: str,
        back_en: str,
        lang: str,
        segment_id: str,
    ) -> BackTranslationVerdict:
        """
        English-only LLM judge comparing original English against back-translated English.
        """
        res = await self.llm_client.judge_back_translation(
            source_en=source_en,
            back_en=back_en,
            lang=lang,
            segment_id=segment_id,
        )
        return BackTranslationVerdict(verdict=res.verdict, evidence=res.evidence)

    async def translate_segment(
        self,
        seg: SegmentSlot,
        target_lang: str,
        glossary: Glossary,
    ) -> TranslatedSegment:
        """
        Executes the translation retry ladder on a single segment.
        Never falls back to source English text on translation failure.
        """
        segment_id = seg.id
        source_text = seg.text

        # Target language code
        target_code = target_lang
        if self.code_mixed:
            info = self.registry.get_info(target_lang)
            if info and info.code_mixed_target:
                target_code = info.code_mixed_target

        terms_to_protect = glossary.get_terms_for_text(source_text)
        canonical_map = {
            t.lower(): glossary.get_canonical(t, target_lang) for t in terms_to_protect
        }

        # Step 1: Translate with placeholders
        protected_text, ph_map = protect_terms(source_text, terms_to_protect)

        last_text: Optional[str] = None
        last_failure_reason: Optional[str] = None

        for attempt in range(1, 4):
            use_placeholders = attempt == 1 and bool(ph_map)
            text_to_translate = protected_text if use_placeholders else source_text

            try:
                raw_translation, _ = await self.bodhan_client.translate(
                    text=text_to_translate,
                    target_language_code=target_code,
                    source_language_code="en",
                    stage="translate",
                    lang=target_lang,
                    segment_id=segment_id,
                )
            except (BudgetExceededError, CircuitBreakerOpen):
                raise
            except Exception as e:
                self.logger.warning(
                    f"Translation call failed on attempt {attempt}: {e}",
                    stage="translate",
                    lang=target_lang,
                    segment_id=segment_id,
                    attempt=attempt,
                )
                last_failure_reason = f"api_error: {e}"
                continue

            # Restore placeholders if used
            if use_placeholders:
                restored_text, survived = restore_terms(raw_translation, ph_map, canonical_map)
                if not survived:
                    self.logger.info(
                        "Placeholders did not survive translation cleanly; falling back to un-placeholder translation: term_unlocked",
                        stage="translate",
                        lang=target_lang,
                        segment_id=segment_id,
                    )
                    last_failure_reason = "placeholder_mismatch"
                    continue
                final_text = restored_text
            else:
                final_text = raw_translation

            # Check if final_text has any Indic characters in expected script
            is_script_ok, _ = check_dominant_script(final_text, target_lang, self.registry)
            if is_script_ok:
                last_text = final_text

            # Run structural validation checks
            checks_pass, failure_reason = run_translation_structural_checks(
                source_text=source_text,
                target_text=final_text,
                target_lang=target_lang,
                registry=self.registry,
            )

            if checks_pass:
                # Step: Back-translation judge
                try:
                    back_en, _ = await self.bodhan_client.translate(
                        text=final_text,
                        target_language_code="en",
                        source_language_code=target_lang,
                        stage="translate",
                        lang=target_lang,
                        segment_id=segment_id,
                    )
                    judge_result = await self.judge_back_translation(
                        source_en=source_text,
                        back_en=back_en,
                        lang=target_lang,
                        segment_id=segment_id,
                    )

                    if judge_result.verdict in ["mistranslation", "omission"]:
                        if attempt < 3:
                            self.logger.warning(
                                f"Back-translation judge flagged '{judge_result.verdict}': {judge_result.evidence}. Paraphrasing English source and re-translating...",
                                stage="translate",
                                lang=target_lang,
                                segment_id=segment_id,
                            )
                            source_text = await self.paraphrase_source_english(
                                source_text, target_lang, segment_id, terms_to_protect
                            )
                            last_failure_reason = f"judge_{judge_result.verdict}"
                            continue
                        else:
                            # Last attempt flagged by judge: must end as translation_flagged
                            return TranslatedSegment(
                                id=segment_id,
                                index=seg.index,
                                source_text=seg.text,
                                target_text=final_text,
                                target_lang=target_lang,
                                status="translation_flagged",
                                flag_reason=f"judge_{judge_result.verdict}: {judge_result.evidence}",
                                back_translation=back_en,
                                judge_verdict=judge_result.verdict,
                                judge_evidence=judge_result.evidence,
                                attempt=attempt,
                            )

                    return TranslatedSegment(
                        id=segment_id,
                        index=seg.index,
                        source_text=seg.text,
                        target_text=final_text,
                        target_lang=target_lang,
                        status="ok",
                        back_translation=back_en,
                        judge_verdict=judge_result.verdict,
                        judge_evidence=judge_result.evidence,
                        attempt=attempt,
                    )
                except (BudgetExceededError, CircuitBreakerOpen):
                    raise
                except Exception as e:
                    self.logger.debug(
                        f"Back-translation check skipped due to error: {e}",
                        stage="translate",
                        lang=target_lang,
                    )
                    return TranslatedSegment(
                        id=segment_id,
                        index=seg.index,
                        source_text=seg.text,
                        target_text=final_text,
                        target_lang=target_lang,
                        status="ok",
                        attempt=attempt,
                    )

            # Checks failed on this attempt
            last_failure_reason = failure_reason
            self.logger.warning(
                f"Translation check failed on attempt {attempt}: {failure_reason}",
                stage="translate",
                lang=target_lang,
                segment_id=segment_id,
                attempt=attempt,
            )

            # Step 3: Paraphrase source text with LLM
            if attempt == 2:
                source_text = await self.paraphrase_source_english(
                    source_text, target_lang, segment_id, terms_to_protect
                )

        # All attempts exhausted
        if last_text and last_text.strip():
            self.logger.warning(
                f"All translation attempts exhausted for segment {segment_id}. Accepting with flag: translation_flagged",
                stage="translate",
                lang=target_lang,
                segment_id=segment_id,
            )
            return TranslatedSegment(
                id=segment_id,
                index=seg.index,
                source_text=seg.text,
                target_text=last_text,
                target_lang=target_lang,
                status="translation_flagged",
                flag_reason=last_failure_reason or "validation_failed",
                attempt=3,
            )
        else:
            self.logger.error(
                f"Translation failed for segment {segment_id}: no valid target text produced. Setting translation_failed.",
                stage="translate",
                lang=target_lang,
                segment_id=segment_id,
            )
            return TranslatedSegment(
                id=segment_id,
                index=seg.index,
                source_text=seg.text,
                target_text="",
                target_lang=target_lang,
                status="translation_failed",
                flag_reason=last_failure_reason or "no_target_text",
                attempt=3,
            )

    async def run(
        self,
        segments: List[SegmentSlot],
        target_lang: str,
        glossary: Glossary,
    ) -> List[TranslatedSegment]:
        """Translates all segments for a target language in parallel bounded by semaphore."""
        async def _translate(seg: SegmentSlot) -> TranslatedSegment:
            return await self.translate_segment(seg, target_lang, glossary)

        return list(await asyncio.gather(*[_translate(seg) for seg in segments]))
