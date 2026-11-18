# Design Notes (DESIGN_NOTES.md)

This document explains the architectural principles, algorithmic decisions, LLM boundaries, and verification strategies employed in `bodhan_dub`.

---

## 1. Role and Boundaries of the LLM

A foundational constraint of the system is that **the LLM never writes, translates, or edits Indian-language text**. Indic linguistic processing belongs strictly to Bodhan's specialized models (Indic-Transcribe, Indic-Translate, Indic-Speak). The LLM functions purely as an English-only reasoning and editorial agent.

### Where the LLM is Used:
1. **Sentence Boundary Refinement (`stages/segment.py`):**
   - *Why:* Pure punctuation splitting can break complex compound sentences unnaturally.
   - *Constraint:* The LLM returns ONLY integer merge indices `[[0, 1], [3, 4]]`. It cannot alter, reword, or drop any transcribed English words. If JSON parsing or validation fails, deterministic punctuation-based merging is used.
2. **Terminology Extraction (`stages/glossary.py`):**
   - *Why:* To identify technical terms, proper nouns, acronyms, and domain concepts for translation protection.
   - *Constraint:* Every extracted term is deterministically verified against the transcript. Any term not present in the source transcript is immediately rejected.
3. **Translation Quality Judge (`stages/translate.py`):**
   - *Why:* To catch subtle semantic drift, hallucinations, or critical omissions without human-in-the-loop.
   - *Constraint:* The LLM compares the original English source with an **Indic-Translate back-translation** into English. The prompt contains zero Indic characters.
4. **Source English Paraphrasing & Condensation (`stages/fit.py`):**
   - *Why:* When a translation repeatedly fails structural checks or exceeds the time slot by >20%, simplifying or condensing the English source creates a more concise Indic translation.
   - *Constraint:* All numbers, dates, and named entities must be preserved verbatim. The output is guarded by the Unicode policy filter.

### Structural Enforcement (Policy Guard):
- **Unicode Rejection:** Every LLM response is checked character-by-character. If any character falls into Arabic or Indic Unicode ranges (U+0600–U+06FF, U+08A0–U+08FF, U+0900–U+0DFF, U+1C50–U+1C7F, U+ABC0–U+ABFF), a `PolicyViolation` exception is raised and the response is discarded.
- **Provider-Agnostic JSON:** All LLM queries use JSON-in-text prompts with Pydantic validation and 1 automatic repair retry, avoiding vendor-locked structured outputs.

---

## 2. Speech-to-Text (ASR) and Chunk Planning

- **VAD Engine Hierarchy:** Primary detection uses Silero-VAD via `onnxruntime` with an automatic, zero-crash fallback to ffmpeg `silencedetect`.
- **Chunk Planning:** FastConformer models accept up to 30.0s of audio. To account for internal acoustic padding, chunk boundaries are capped at a **hard ceiling of 29.0s**, targeting 25.0s cuts inside the nearest natural silence interval.
- **Dual-Mode Timestamps:** Requesting `verbose_json` from Indic-Transcribe yields word- and segment-level timestamps. These timestamps are snapped against VAD speech intervals. If coarse timestamps are returned, sentence bounds are proportionally allocated within active VAD speech regions.
- **Hallucination Filtering:** Segments with `<40%` time overlap with VAD speech (`no_speech_overlap`) or 3+ repeated n-grams (`asr_loop`) are discarded.

---

## 3. Speech Synthesis (TTS) and Dynamic Calibration

- **Punctuation Splitting Threshold:** Indic-Speak server audio truncates at ~30.55s. Any translated sentence exceeding **180 characters** is split at clause delimiters (`।`, `.`, `?`, `,`) using grapheme-safe regex `\X`. Audio pieces are stitched using 10ms smooth crossfades.
- **Dynamic CPS Calibrator (`CPSCalibrator`):** Characters-per-second varies across voices (e.g., Kavya ~11.4 cps, Sourav ~13.8 cps). The calibrator tracks a running median of the last 15 successful synthesis calls per voice to predict audio duration before calling TTS.
- **24kHz to 48kHz Resampling:** Synthesized 24kHz mono audio is resampled to 48kHz float32 using 2x polyphase interpolation (`resample_poly`) to match standard broadcast containers.

---

## 4. Acoustic QA and the 3-Attempt Retry Ladder

Every synthesized audio segment undergoes back-transcription using Indic-Transcribe to compute Character Error Rate (CER):
- **CER < 0.15:** Clean pass.
- **0.15 ≤ CER < 0.30:** Warning; accepted if attempts exhausted.
- **CER ≥ 0.30:** Failure; triggers retry planner.

The QA engine also detects acoustic defects:
1. **Clipping:** Saturated samples (`max ≥ 0.999`).
2. **Internal Silence:** Unnatural gaps lasting >1.5s inside speech.
3. **Tail Omission:** Truncation of the final ~15% of the sentence.
4. **Repetitions:** Autoregressive looping of syllables.

### Retry Actions:
- Attempt 1: Retry synthesis with `split_text` or `retry_same`.
- Attempt 2: Paraphrase source English for clearer pronunciation.
- Attempt 3: Condense source English to reduce syllable load.
- Exhaustion: Pick candidate with lowest CER (flag `degraded`), or if unusable (`CER ≥ 0.45` or silent), flag `skipped` and preserve original audio for that slot.

---

## 5. Duration Fitting and Timeline Scheduling

The pipeline enforces strict temporal synchronization:
- **Ratio ≤ 1.0:** Keep natural speed (1.0x), pad silence in scheduler.
- **1.0 < Ratio ≤ 1.20:** Pitch-preserving time stretch with ffmpeg `atempo`.
- **Ratio > 1.20:**
  1. Borrow leading slack (up to 250ms earlier into `slack_before`).
  2. Borrow trailing slack into `slack_after`.
  3. Clamp time stretch strictly at **1.20x max** (never speed up beyond 1.20x to prevent chipmunk effects).
  4. Flag `overrun` if speech still exceeds the extended slot.
- **Zero-Overlap Guarantee:** Segments are placed on the timeline at `placed_start = max(target_start, last_placed_end)`.
- **Sync Bound:** Warnings are logged if sync drift exceeds the **300ms SLA target**.

---

## 6. Mixing, Loudness, and Video Multiplexing

- **Dialogue Ducking:** Original English speech is muted with a **40ms linear crossfade** envelope. This removes original dialogue cleanly while keeping background music, room tone, and sound effects audible. Skipped segment slots remain un-muted.
- **EBU R128 Loudness Normalization:** Mixed audio is measured using `pyloudnorm` and normalized to **−16.0 LUFS integrated (±1.0 LUFS)** with a true-peak ceiling at **−1.0 dBTP**.
- **Multiplexing (`media/mux.py`):** Video and audio are combined using `-c:v copy` (or H.264 fallback), AAC 192kbps audio, and `-movflags +faststart`.
- **Post-Mux Verification:** An automated ffprobe check verifies that video and audio durations match within **±100ms**. If duration drift is detected, bounded padding/trimming repair is executed (max 2 attempts).
