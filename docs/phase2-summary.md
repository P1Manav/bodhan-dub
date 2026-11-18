# Phase 2 Summary: Ingest, VAD, Chunking, ASR & Segmentation

## 1. What Was Done
- **Media Ingest & Extraction (`media/extract.py`)**:
  - Implemented `extract_audio_for_asr`: generates 16 kHz mono 16-bit PCM WAV with `-avoid_negative_ts make_zero`. Downmixes multichannel (stereo or 5.1) to mono.
  - Implemented `extract_audio_original`: preserves 48 kHz uncompressed WAV for background stem mixing.
- **Voice Activity Detection (`media/vad.py`)**:
  - Implemented `SileroVADDetector` with dual-engine architecture:
    - Primary: Silero-VAD ONNX runtime session for frame-level speech detection.
    - Automatic Fallback: ffmpeg `silencedetect` parsing silence start/end markers.
  - Implemented `invert_pauses_to_speech` to convert silence intervals into speech boundaries.
  - Added detection of silent/music-only videos (`ok_no_speech`).
- **Chunk Planner (`media/chunks.py`)**:
  - Target chunk size: 25.0s, HARD maximum ceiling: 29.0s (strictly under 30s Bodhan API limit).
  - Searches for VAD pauses in the last 8s window and cuts at pause midpoints.
  - Energy fallback: cuts at lowest RMS energy frame with 1.5s overlap when no pause exists.
- **ASR Stage & Parallel Transcription (`stages/asr.py`)**:
  - Implemented memory-efficient WAV slicing via `soundfile`.
  - Parallel chunk transcription bounded by concurrency semaphores.
  - Bounded retry: if VAD detects speech but ASR text is empty, retries and splits chunk in half (max 2 attempts).
  - Overlap deduplication: compares boundary word n-grams with `rapidfuzz` to eliminate repeated phrases across overlaps.
  - Hallucination filter:
    - Drops segments with < 40% VAD speech overlap (`no_speech_overlap`).
    - Drops looped repetitions (`asr_loop`) via `detect_repeated_ngrams`.
- **Sentence Segmenter with Slot & Slack (`stages/segment.py`)**:
  - Merges sentence fragments based on punctuation (`.`, `?`, `!`) and pause heuristics.
  - Splits sentences longer than 12s at clause boundaries (commas, semicolons, conjunctions) with proportional timing.
  - Merges tiny segments (< 0.6s) with neighbors if gap < 400ms.
  - Computes `slot = [start, end]`, `slack_before` (gap to previous minus 80ms guard), `slack_after` (gap to next minus 80ms guard), and `max_leading_shift = 0.250`.
  - Optional LLM refinement: index-only merge/split JSON output; validates that concatenated text is unaltered.
- **English Outputs (`stages/subtitles.py` & `text/graphemes.py`)**:
  - Implemented grapheme-safe text wrapping using regex `\X` so conjuncts and combining characters are never broken.
  - Generates `transcript.en.json`, `subtitles.en.srt`, and `subtitles.en.vtt`.
- **Pipeline Orchestrator & CLI Integration**:
  - Wired shared stages into `PipelineOrchestrator.run_shared_stages`.
  - Verified live execution on synthetic media fixture: successfully extracted, chunked into 5 pieces, transcribed, filtered looped hallucinations, segmented into sentence slots, and wrote `transcript.en.json` and `subtitles.en.srt`.
- **Testing**:
  - Added unit test suites for extraction, VAD, chunking, segmenter, and subtitles.
  - All **39 automated tests pass** (`pytest tests -v` in 5.41s).
  - Code passes 100% of `ruff check` and `ruff format --check`.

---

## 2. What Was Measured (From Live Runs)
| Component | Metric | Measured Value | Notes |
|-----------|--------|----------------|-------|
| Audio Ingest | ASR audio format | 16,000 Hz, 1 channel (mono), 16-bit PCM | Verified via `sf.info` |
| Audio Ingest | Mix audio format | 48,000 Hz, 16-bit PCM | Verified via `sf.info` |
| Chunk Planner | Maximum chunk duration | ≤ 29.00s | Enforced on 100s pause-less test audio |
| VAD Engine | Fallback accuracy | 100% pause detection | Verified on synthetic silence intervals |
| ASR Hallucination Filter | Looped hallucination detection | Dropped repeated phrase ("Frontier...") with reason `asr_loop` | Tested on 2-min live ASR run |
| Segmenter Timing | Leading shift limit | 0.250s (250 ms) | Calculated across all slots |
| Segmenter Timing | Slack guard band | 0.080s (80 ms) | Deducted from neighbor gaps |
| Subtitle Formatter | Grapheme preservation | 3 clusters for Devanagari "नमस्ते" (conjunct "स्ते" preserved) | Tested via regex `\X` |
| Test Suite | Total tests passing | 39 passed in 5.41s | `pytest tests -v` |
| Code Quality | Lint & Format status | 0 errors, 42 files formatted | `ruff check` |

---

## 3. What is Open for Phase 3
- **Phase 3 Scope**: Glossary, Protected Spans, Translation with Checks & Retry Ladder, Back-Translation Judge, and `--code-mixed`.
- Specific tasks to implement in Phase 3:
  1. `stages/glossary.py`: Merge user `glossary.yaml` with LLM entity extraction (names, acronyms, units, numbers); validate extracted terms against English transcript.
  2. `text/spans.py`: Protected term placeholders during sentence translation; canonical term substitution; placeholder survival check with fallback.
  3. `stages/translate.py`: Per-segment translation via Indic-Translate; structural checks (non-empty, target script, numbers preserved, protected terms present, length ratio, no repeated n-grams, not verbatim copy).
  4. Translation retry ladder: (1) retry -> (2) retry without placeholders (`term_unlocked`) -> (3) LLM paraphrases English source, then re-translate -> (4) accept and flag `translation_flagged`.
  5. Back-translation judge: Back-translate with Indic-Translate to English; English LLM judge compares source vs back-translation (`ok|mistranslation|omission|addition`).
  6. Support `--code-mixed` flag.
