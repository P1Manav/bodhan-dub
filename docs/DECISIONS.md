# Architectural Decisions (DECISIONS.md)

This log records design decisions made during development of `bodhan_dub`, following the format: `[Date] Decision: Reason`.

---

### [2026-11-18] DEC-001: Model Selection for Indic-Transcribe
- **Decision:** Use `indic-transcribe` (Core FastConformer architecture).
- **Reason:** The hosted Bodhan API key specifically enforces `models=['indic-transcribe']` and returns HTTP 403 on `indic-transcribe-flex`. Furthermore, Core provides superior Word Error Rate (8.9 vs 11.1 for Flex) on formal transcription.

### [2026-11-18] DEC-002: Dual-Mode Timestamp Strategy (Verbose JSON + VAD Snapping)
- **Decision:** Request `response_format="verbose_json"` from `indic-transcribe` to obtain segment/word timestamps, but always snap and validate against Silero-VAD speech boundaries. If timestamps are coarse or absent, fall back to proportional VAD pause allocation.
- **Reason:** While live probe confirmed `verbose_json` returns `segments`, ASR timestamps in noisy or background-music conditions can drift or hallucinate outside speech boundaries. VAD provides frame-accurate speech bounds.

### [2026-11-18] DEC-003: Indic-Speak Input Splitting Threshold
- **Decision:** Split synthesized text at clause/sentence boundaries (`।`, `.`, `?`, `,`) using grapheme-safe regex `\X` so no individual TTS request exceeds 180 characters.
- **Reason:** Live probe measured that the Indic-Speak server truncates audio output at ~30.55 seconds (around 240+ characters). Splitting at ≤180 chars ensures zero server-side truncation and predictable duration fitting.

### [2026-11-18] DEC-004: Target Language Code Conventions
- **Decision:** Use standard ISO 639-1 / script codes (`as, bn, en, gu, hi, kn, ml, mr, or, pa, ta, te, ur`, etc.) as verified by the API's error feedback.
- **Reason:** Passing descriptive names like `"Hindi (code-mixed)"` returns HTTP 400 with a list of supported codes. Code-mixing is handled within the translation context and vocabulary.

### [2026-11-18] DEC-005: 24kHz to 48kHz Audio Resampling
- **Decision:** Indic-Speak returns 24 kHz mono audio. The mixing pipeline operates strictly in float32 at 48 kHz. All synthesized stems are resampled using high-quality 2x polyphase interpolation or scipy/librosa/soundfile.
- **Reason:** Professional video dubbing delivery requires 48 kHz audio to match standard video container specifications (ITU-R BS.1770 / EBU R128).

### [2026-11-18] DEC-006: LLM Structural Guard via Unicode Rejection
- **Decision:** Enforce that LLM inputs and outputs NEVER contain Indian script or Arabic-derived script characters (Unicode ranges U+0600–U+06FF, U+0900–U+0DFF, U+1C50–U+1C7F). If an LLM call produces characters in these ranges, the response is discarded and deterministic fallback is triggered.
- **Reason:** Hard constraint: the LLM must only analyze and rewrite English. Indian language generation is reserved exclusively for Bodhan Indic-Translate and Indic-Speak.

### [2026-11-18] DEC-007: VAD Engine Hierarchy (Silero ONNX with ffmpeg Fallback)
- **Decision:** Primary VAD uses Silero-VAD via onnxruntime. If model weights are missing or onnxruntime encounters any platform issue, the system automatically falls back to ffmpeg `silencedetect`.
- **Reason:** Guarantees zero runtime crashes across all environments while retaining high-precision VAD speech regions when onnxruntime is loaded.

### [2026-11-18] DEC-008: Hard Maximum 29.0s Chunk Ceiling
- **Decision:** Cap chunk duration strictly at 29.0s (1.0s buffer below Bodhan's 30.0s API threshold).
- **Reason:** FastConformer acoustic models require internal sample padding. Submitting 30.0s exact audio can exceed the server's duration threshold by milliseconds and return HTTP 400.

### [2026-11-18] DEC-009: Hallucination Filtering Criteria
- **Decision:** Drop transcribed segments that have < 40% time overlap with active VAD speech regions (`no_speech_overlap`), or that exhibit 3+ consecutive repetitions of any 1-to-4-word n-gram (`asr_loop`).
- **Reason:** ASR models frequently hallucinate repetitive loops or phantom phrases on background music, noise, or silence. This filter stops hallucinations before translation.

### [2026-11-18] DEC-010: Duration Fit Policy and atempo Stretcher
- **Decision:** Audio with duration ratio <= 1.0 is kept at natural speed (1.0x) and padded with silence. Ratios in (1.0, 1.20] are time-stretched using ffmpeg `atempo`. For ratios > 1.20, borrow slack (up to 250ms before, plus trailing slack), cap stretch strictly at 1.20x, and flag `overrun` if speech still exceeds the extended slot. Never speed up beyond 1.20x, never slow down below 1.0x by default, and never truncate speech.
- **Reason:** Maintains natural, professional vocal delivery without unnatural chipmunk effects or abrupt speech truncation.

### [2026-11-18] DEC-011: Acoustic Speech QA and Intelligent Retry Loop
- **Decision:** Every synthesized segment is back-transcribed with Indic-Transcribe to compute Character Error Rate (CER), detect repeated n-grams, and detect tail truncation/omissions. Retries are planned dynamically (`retry_same`, `split_text`, `paraphrase_source`, `condense`) and bounded to 3 attempts. If attempts are exhausted, the candidate with the lowest CER is selected and flagged `degraded`; if unusable (CER ≥ 0.45 or empty/silent), the slot is flagged `skipped` and original audio is retained.
- **Reason:** Guarantees acoustic and textual fidelity while preventing pipeline stalls on difficult phonetic passages.

### [2026-11-18] DEC-012: Timeline Scheduler and Collision Prevention
- **Decision:** Schedule fitted segments sample-accurately onto the 48kHz master audio timeline. Place segments at `placed_start = max(target_start, last_placed_end)` to eliminate all overlaps, and log a warning if drift exceeds the 300ms sync target.
- **Reason:** Prevents speech collisions and maintains synchronization with on-screen speaker lip movements and scene cuts.

### [2026-11-18] DEC-013: Loudness Staging and Muting Crossfades
- **Decision:** Mute original speech regions with 40ms linear fade-out and fade-in envelopes to preserve ambient sound and background music while removing original English dialogue. Normalize the final mixed audio to -16.0 LUFS integrated (±1.0 LUFS) with a -1.0 dBTP true-peak ceiling.
- **Reason:** Complies with ITU-R BS.1770 and EBU R128 international broadcast and streaming audio standards.

### [2026-11-18] DEC-014: AI-Dubbed Metadata and Output Verification
- **Decision:** Multiplex video and dubbed audio into MP4 with `-movflags +faststart`, MP4 comment metadata `"AI-dubbed with Bodhan Indic-Speak"`, and ISO audio track language tags. Verify via ffprobe that audio and video stream durations match within ±100ms, triggering bounded padding or trimming repair (max 2 attempts) if drift is detected.
- **Reason:** Transparent AI labeling per ethical standards and seamless playback across web browsers and media players without desync.
