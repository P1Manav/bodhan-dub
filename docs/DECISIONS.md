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

### [2026-11-21] DEC-015: Post-synthesis Condensation Loop and Entity Preservation Guard
- **Decision:** When synthesized speech duration exceeds `1.20 * slot_duration`, execute up to 2 post-synthesis English condensation attempts with an LLM character budget calculated from `slot_duration * calibrator.get_cps(lang, voice)`. Require that numbers, acronyms, capitalized names, and glossary terms are deterministically verified in the condensed output. Re-translate with Indic-Translate, re-synthesize, and keep the best candidate by CER then fit ratio.
- **Reason:** Resolves syllable expansion before relying on slack borrowing and time stretching, preventing unnatural speech rates while guaranteeing factual accuracy and entity preservation.

### [2026-11-21] DEC-016: Non-cascading Sync Budget Placement with Lookahead
- **Decision:** Timeline placement computes start times in a single pass with lookahead. If a segment exceeds its slot, borrow following slack and stretch up to 1.20x; if still overflowing, flag `overrun` and accept local drift rather than cascading desync onto downstream segments. At media boundary, shift earlier up to 250ms; if speech still exceeds file duration, flag `truncation_risk` and degrade segment without silently truncating audio.
- **Reason:** Prevents desync propagation across multi-minute videos and ensures speech is never dropped or silently cut.

### [2026-11-21] DEC-017: Multi-factor Speech QA Gating with Configurable Thresholds
- **Decision:** Expand QA gating to fail segments exhibiting internal silence >1.5s or duration-vs-prediction outside [0.6, 1.6]. Make CER thresholds configurable (`cer_warn=0.15`, `cer_fail=0.30`, `cer_degraded_max=0.45`). When strict QA fails on all 3 attempts, retain the candidate with lowest CER and mark `degraded` if CER ≤ `cer_degraded_max`, or `skipped` if above.
- **Reason:** Keeps the pipeline honest and resilient: partial phonetic degradation is accepted with clear reporting, while severe acoustic flaws or speech dropouts are skipped to preserve original audio.

### [2026-11-21] DEC-018: Voice-Agnostic Dynamic CPS Calibration
- **Decision:** Remove all hard-coded language/voice seeds from `CPSCalibrator`. Initialize unmeasured voices at generic default (12.0 cps) or from `languages.yaml` hints, and update running history and EWMA (`alpha=0.3`) dynamically after each synthesized segment.
- **Reason:** Eliminates bias, supports arbitrary voices or languages, and smoothly converges to actual speech rates.

### [2026-11-21] DEC-019: Polyphase True-Peak Limiting and Equal-Power Ducking
- **Decision:** Implement 4x polyphase oversampling via `scipy.signal.resample_poly` to detect and clamp inter-sample true peaks to -1.0 dBTP. Apply 40ms equal-power (`cos`/`sin`) gain crossfades to duck original speech regions while leaving skipped slots completely unmuted.
- **Reason:** Prevents digital clipping distortion upon lossy AAC encoding and preserves broadcast loudness compliance (-16.0 ± 1.0 LUFS) with seamless ambient continuity.

### [2026-11-21] DEC-020: Format-Preserving Video Muxing with Duration Lock
- **Decision:** Ensure muxing passes `-t <input video duration>` to match input container length exactly, remove `-shortest`, write clean metadata without escaped inner quotes, handle non-zero input `start_time` with `-itsoffset`, and fall back to `-c:v libx264 -pix_fmt yuv420p` if stream copy fails. Enforce strict verification within ±100ms.
- **Reason:** Eliminates audio/video drift across variable frame rate or rotated video files while preserving full format compatibility.

### [2026-11-21] DEC-021: LLM Client Hardening and Prompt Injection Isolation (<data> wrapper & Host Allowlist)
- **Decision:** All LLM requests are guarded by `HostAllowlistGuard` on outgoing traffic, failing fast in `__init__` on unauthorized hosts. Untrusted audio transcript text and user input are strictly wrapped in `<data>...</data>` with explicit system instructions prohibiting command execution. Implemented Decision Point #6 (`plan_retry_action`) choosing among enum `['split_text', 'paraphrase_source', 'condense', 'retry_same']` with deterministic fallback, and Decision Point #7 (`extract_glossary_terms`) with strict transcript substring validation, number/punctuation filtering, and length-descending sort.
- **Reason:** Prevents prompt injection attacks embedded in video transcripts from hijacking LLM linguistic editing tasks, and ensures 100% deterministic reliability even if the LLM fails or is unavailable.

### [2026-11-21] DEC-022: Isolated Circuit Breakers per (Service, Language) Tuple
- **Decision:** Manage circuit breakers via `CircuitBreakerRegistry` holding isolated state per `(service, language)` tuple. Trips on 5 consecutive 5xx failures per service/language. In `execute_with_resilience`, the concurrency semaphore is released prior to sleeping on 429/5xx backoff.
- **Reason:** An outage in Tamil TTS does not trip Hindi ASR or Telugu TTS, and releasing semaphores during backoff prevents slot starvation across concurrent tasks.

### [2026-11-21] DEC-023: Full End-to-End Language and Segment Concurrency
- **Decision:** Run per-language processing in parallel up to `--concurrency` (default 2), and parallelize per-language segment translation (bounded by `semaphore_translate`) and segment synthesis (bounded by `semaphore_tts`). Synchronize `CPSCalibrator` updates across concurrent synthesis tasks with `asyncio.Lock`.
- **Reason:** Provides significant wall-clock speedup when dubbing into multiple languages while respecting server rate limits and eliminating race conditions.

### [2026-11-21] DEC-024: CLI Operational Flags Wiring
- **Decision:** Fully wire all operational flags through `cli.py` and `orchestrator.py`: `--concurrency` (default 2), `--voices` (custom YAML/JSON mapping `lang: voice`), `--resume` (checks `manifest.json` completed segments and intermediate WAVs to avoid redundant synthesis), `--separate-music` (checks for demucs/spleeter and falls back cleanly to VAD ducking if missing), and `--keep-work` (cleans up intermediate work dir on clean exit by default, retains if flag is set).
- **Reason:** Provides full operator controllability, robust crash recovery, and seamless intermediate asset inspection.

### [2026-11-21] DEC-025: Silero VAD Local Asset Integrity and Cryptographic Hash Check
- **Decision:** Verify SHA256 of `silero_vad.onnx` against pinned hash `2ea52a1c618ec50428f5370d0b8b1a8d0f1b2123d47d953a99e71e74a80fb4ff` across explicit, asset, package, and root paths. If missing or corrupt, log a clear warning and fall back cleanly to ffmpeg `silencedetect` without crashing.
- **Reason:** Protects against poisoned or corrupt ONNX models while ensuring zero crashes on bare environments.

### [2026-11-21] DEC-026: Automatic Speaker Voice Pitch Clustering
- **Decision:** When `--speaker-voices auto` is set, estimate fundamental frequency (F0) per speech segment via autocorrelation; cluster segments into male/female (or 2 clusters), and map clusters to male and female voices from `LanguageRegistry.get_default_voice`. Fallback to default voice if clustering fails or single speaker.
- **Reason:** Supports multi-speaker videos with automatic vocal identity preservation without requiring external speaker diarization models.

### [2026-11-21] DEC-027: Reusable Dubbing Analyzer and Empirical Baseline Protocol
- **Decision:** Implement `scripts/analyze_dub.py` using strictly non-ML signal processing tools (`ffmpeg`, `ffprobe`, `numpy`, `pyloudnorm`) without any non-Bodhan ASR. Evaluates: (1) container format and stream duration deltas (pass $\le 100\text{ ms}$); (2) EBU R128 integrated LUFS and 4x true peak; (3) 100ms energy VAD onset alignment, p95/max offset, and ghost speech quantification; (4) sliding 1s waveform correlation cross-referenced with `segments.<lang>.json` skipped slots; (5) long silence (>3s) narration drop detection; (6) report.json honesty verification. Store baseline empirical findings in `docs/VIDEO_ANALYSIS.md`.
- **Reason:** Provides an objective, reproducible, model-free regression measurement harness to detect timing drift, loudness discrepancies, dropped narration, and mute leakage.

### [2026-11-21] DEC-028: Lookahead True-Peak Limiter and Multi-Stage Loudness Iteration (B3)
- **Decision:** Implement lookahead true-peak limiting in `media/loudness.py` with 4x polyphase oversampled peak detection, 5ms lookahead delay, 5ms attack / 50ms release envelope follower, and hard ceiling at -1.0 dBTP. Replace single-pass gain reduction with multi-stage loudness pipeline: normalize segments -> normalize speech stem to -16 LUFS -> mix with background ducking -> apply true-peak limiter -> measure final mix and iterate gain up to 3 times to guarantee both -16 ± 1 LUFS and $\le -1.0\text{ dBTP}$. Enforce loudness and audio-stream presence checks in `media/mux.py::mux_and_verify`.
- **Reason:** Direct gain cuts previously deformed audio loudness after normalization, failing EBU R128 targets. Lookahead envelope limiting preserves audio dynamics while strictly preventing digital inter-sample clipping.

### [2026-11-21] DEC-029: Single-Stretch from Raw Synthesis and Non-Cascading Drift Budget (B6 & B13)
- **Decision:** Store raw unstretched synthesis audio on `SegmentFitResult` and ensure `TimelineScheduler` always re-stretches from the raw synthesized audio rather than re-stretching previously stretched buffers. Hard-cap cumulative speed factor at 1.20x max. In `TimelineScheduler`, enforce non-cascading drift: each segment anchors to its source start within a strict $\pm 300\text{ ms}$ window and does not push subsequent segments into the future if an earlier segment overruns. Overlaps are resolved via preceding slack borrowing, 1.20x stretch, following slack borrowing, and $\le 100\text{ ms}$ crossfade with `overlap_forced` flag.
- **Reason:** Cascading offsets previously caused 11.4s of acoustic drift across the video. Non-cascading scheduling with single-pass stretch preserves voice naturalness and guarantees tight alignment.

### [2026-11-21] DEC-030: Mute Mask Union of Source Speech and Placed Audio with Holes for Skipped Slots (B11)
- **Decision:** In `media/mix.py`, compute the mute mask as the mathematical union of all detected source speech regions and all placed dubbed segment spans, applied with 40ms equal-power cosine crossfades. Punch explicit holes in the mute mask for skipped segments (`[start, end]`) so original English audio remains audible only for intentionally skipped slots. Clamp placed dubbed spans to $[start - 300\text{ms}, end + 300\text{ms}]$ to eliminate ghost speech.
- **Reason:** Eliminates the 14s of original English audio leakage found during baseline analysis and prevents dubbed speech from playing in empty silence.

### [2026-11-21] DEC-031: Event Loop Offloading and Vectorized Signal Processing (B8)
- **Decision:** Wrap all blocking CPU/IO operations (atempo time stretching, ffmpeg silence detection, crossfading, Demucs separation, and disk reads/writes $>100\text{ KB}$) in `asyncio.to_thread`. Vectorize `detect_internal_silence` and `crossfade_concat` in `media/audio_proc.py` using NumPy vectorized operations and run-length encoding instead of Python sample loops.
- **Reason:** Prevents event loop blocking during multi-language processing, ensuring concurrent language pipelines run smoothly without jitter.

### [2026-11-21] DEC-032: Warm-Up Calibration and Post-Synthesis Condensation (B7)
- **Decision:** Warm up `CPSCalibrator` by sequentially synthesizing the first $N=3$ segments of a language before predicting duration for remaining segments. In `stages/fit.py`, if synthesized audio exceeds the available slot by $> 1.20\text{x}$ even with slack borrowing, execute up to two post-synthesis English condensation loops via the LLM, re-translate, and re-synthesize before falling back to stretch $\le 1.20\text{x}$ and flagging `overrun`. Remove dead duplicate blocks in candidate selection.
- **Reason:** Eliminates cold-start prediction errors on unseen voice profiles and drastically reduces overrun rates on verbose translations while strictly respecting the 1.20x stretch ceiling.

### [2026-11-21] DEC-033: Grapheme-Aware Punctuation Splitting and Safe Chunk Limits (B10)
- **Decision:** Enhance `split_text_at_punctuation` in `stages/tts.py` to protect decimal numbers (`3.14`), abbreviation dots, and numbers with commas from being split. If punctuation-split chunks still exceed `max_chars` (safe limit from language configuration, default 180), fall back to whitespace splitting, and if a single word exceeds `max_chars`, split cleanly at Unicode grapheme cluster boundaries (`\X`). Remove all `except TypeError` retry blocks.
- **Reason:** Prevents malformed audio chunks and numbers from being severed into gibberish while eliminating fragile exception swallowing.

### [2026-11-21] DEC-034: Reference-Relative Speech QA and Resilient Back-Transcription (B14)
- **Decision:** Include duration mismatch (ratio outside $[0.65, 1.35]$), digital clipping, and internal pause ($>1.5\text{s}$) directly in the QA `passed` boolean determination. Compare detected n-gram repetitions against the reference Indic text: if the repeated pattern exists in the reference text (e.g. natural Indic reduplication such as "बार-बार" or "छोटे-छोटे"), do not flag it as an error. If back-transcription fails due to transient 5xx or API errors, mark the segment as `degraded`/`unverified` (CER 0.0) without burning TTS retry attempts or crashing the pipeline.
- **Reason:** Avoids false positives on valid linguistic reduplication and prevents backend ASR outages from consuming synthesis retries.

### [2026-11-21] DEC-035: Four-Step Translation Ladder and Configurable Latin Character Tolerance (B17)
- **Decision:** Implement a structured 4-step translation ladder: (1) entity placeholders; (2) raw un-placeholdered translation; (3) English paraphrase re-translation; (4) accept last valid Indic candidate with `translation_flagged` (or `translation_failed` with empty target text if no Indic text exists). Allow Latin character share up to 45% to accommodate technical acronyms (e.g. "AI", "NASA") while rejecting pure English. Use dynamic length-ratio bounds based on source sentence length, and wrap parallel translation in `return_exceptions=True` to retain partial results on budget stops.
- **Reason:** Replaces ad-hoc retry logic with a deterministic ladder, avoids rejecting sentences with legitimate technical loanwords, and guarantees partial state retention.

### [2026-11-21] DEC-036: Cross-Platform Process Interruption and Exit Code Contracts (B9, B20)
- **Decision:** Handle SIGINT and SIGTERM across POSIX (via loop signal handlers) and Windows (via `signal.signal`), triggering task cancellation, flushing manifest, partial report, and run log before exiting with code 130. Enforce strict exit code contracts: 0 = success, 1 = failure, 2 = preflight/usage error, 130 = interrupt.
- **Reason:** Guarantees no orphaned tasks or lost state on Ctrl-C and adheres to UNIX exit code standards.

### [2026-11-21] DEC-037: Audio Stream Padding, Codec Compatibility, and Stream Selection in Multiplexing (B12)
- **Decision:** When source audio is shorter than the video container (e.g. the 2s gap in `English.mp4`), pad original audio to exact video duration in `media/extract.py`. In `media/mux.py`, probe video codec and container format, falling back to H.264 CRF 18 re-encoding if stream copy fails. Use explicit stream mapping `-map 0:v:0 -map 1:a:0` and clean up temporary repair WAVs upon verification.
- **Reason:** Resolves audio/video duration mismatch at the end of output videos and handles arbitrary container formats safely.

### [2026-11-21] DEC-038: Cryptographic Content-Addressed Resume and Checkpoint Integrity (B15)
- **Decision:** In `manifest.py` and `orchestrator.py`, key segment checkpoints by the cryptographic hash `sha256(source_text + translation + voice + language)`. Store and restore complete measured metrics (`res.model_dump`) beside cached WAVs, ensuring resume doesn't alter reported statistics or accept stale audio.
- **Reason:** Prevents stale checkpoints from contaminating runs if source text, translation, or voice settings change between runs.

### [2026-11-21] DEC-039: Atomic Cache Writes and Payload Validation (B16)
- **Decision:** In `gateway/cache.py`, validate that cached data is non-empty before writing. Use atomic temporary file writes (`.tmp.<pid>`) and atomic replacement (`os.replace`) to prevent corrupted cache reads across concurrent workers.
- **Reason:** Eliminates race conditions and corrupted disk cache entries under multi-language concurrency.

### [2026-11-21] DEC-040: Pre-Call Budget Reservation and HTTP-Date Retry-After Handling (B18)
- **Decision:** In `gateway/cost.py`, implement `BudgetGuard.reserve` to hold estimated cost before initiating paid API calls and reconcile after completion. In `gateway/resilience.py`, parse `Retry-After` headers supporting both integer seconds and RFC 2822 HTTP-date strings, and record network timeouts toward circuit breaker failure counts (504 Gateway Timeout).
- **Reason:** Prevents concurrent tasks from overshooting the configured budget ceiling and ensures standards-compliant rate limit handling.

### [2026-11-21] DEC-041: Preflight Disk Space, Duration Clamping, and Full Metric Reporting (B20, B21)
- **Decision:** In `preflight.py`, verify available disk space ($\ge 500\text{ MB}$ or $3\times$ video size) and validate LLM settings upfront. In `stages/report.py`, include runtime target, speech stem LUFS, final mix LUFS, and true peak dBTP in `report.json`. Update `README.md` to clarify the 13 production languages (1 source + 12 targets).
- **Reason:** Fails fast before spending resources on full disks or misconfigured environments, and provides complete audio engineering transparency in reports.

---

## Folded Notes: Ingest, VAD, Chunking, ASR & Segmentation (Phase 2 Archive)
- **Audio Extraction**: 16 kHz mono 16-bit PCM for ASR with `-avoid_negative_ts make_zero`; 48 kHz uncompressed WAV for mixing.
- **VAD Dual Engine**: Silero-VAD ONNX with automatic ffmpeg `silencedetect` fallback; pause inversion logic for speech regions.
- **Chunk Planning**: Target 25s, hard max ceiling 29s (under 30s Bodhan limit); pause midpoint splitting with energy fallback.
- **ASR Verification**: Memory-efficient WAV slicing; bounded retry on empty text; overlap deduplication with rapidfuzz; hallucination filtering (<40% VAD overlap or repeated n-grams dropped).
- **Sentence Segmenter**: Sentence boundary merging, clause splitting for segments >12s, tiny segment (<0.6s) merging, slot/slack calculation with 80ms guard band and 250ms max shift.
- **English Subtitles**: Grapheme-safe wrapping (regex `\X`) for `transcript.en.json`, `subtitles.en.srt`, `subtitles.en.vtt`.

