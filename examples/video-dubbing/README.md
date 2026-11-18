# bodhan_dub: Autonomous English-to-Indic Video Dubbing Pipeline

`bodhan_dub` is an autonomous end-to-end Python pipeline that dubs English videos into 13 production-tier Indian languages using **strictly Bodhan Indic AI models** (`indic-transcribe`, `indic-translate`, `indic-speak`).

All generated media is labeled as **AI-dubbed** via MP4 container metadata, structured reports, and subtitles.

---

## Key Features

- **Indic-Only Speech & Translation:** Speech recognition is performed exclusively via `indic-transcribe`, translation via `indic-translate`, and synthesis via `indic-speak`.
- **Strict English-Only LLM Guard:** The LLM never writes, rewrites, or translates Indian-language text. It functions strictly as an English reasoning agent (sentence segmentation, terminology extraction, and back-translation evaluation). A character-level Unicode policy guard rejects non-English characters.
- **Dual-Engine Voice Activity Detection (VAD):** High-precision Silero ONNX speech boundary detection with an automatic, zero-crash fallback to ffmpeg `silencedetect`.
- **Bounded Chunk Planning:** FastConformer 29.0s chunk ceiling targeting natural pauses near 25.0s, with hallucination filtering for music and looped phrases.
- **Dynamic CPS Calibration & Punctuation Splitting:** Synthesized text is split at clause delimiters (≤180 chars) to prevent server audio truncation, and character-per-second rates are calibrated per voice profile.
- **Acoustic Speech QA & Retry Ladder:** Synthesized segments are back-transcribed to evaluate Character Error Rate (CER), detect syllable repetitions, and catch tail omissions, planning bounded retries (same text, split text, paraphrase, condense).
- **Pitch-Preserving Duration Fit:** Natural speed for ratio ≤ 1.0; ffmpeg `atempo` time-stretching (1.0x–1.20x); slack borrowing (up to 250ms earlier + trailing slack) with a hard 1.20x stretch clamp.
- **EBU R128 Loudness Staging & Muxing:** Original dialogue is muted with a 40ms linear crossfade to preserve background music/ambience, mixed to −16.0 LUFS integrated (±1.0 LUFS) with a −1.0 dBTP true-peak ceiling, and verified with ffprobe within ±100ms.

---

## Quickstart

### Prerequisites
- Python 3.11+
- `ffmpeg` (version 7.0+) and `ffprobe` on system PATH

### Installation
```bash
# Clone and enter directory
cd examples/video-dubbing

# Create virtual environment and install exact pinned dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .

# Or using Makefile
make setup
```

### Environment Configuration
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
Required environment variables:
```env
BODHAN_TRANSCRIBE_KEY=sk-...
BODHAN_TRANSLATE_KEY=sk-...
BODHAN_SPEAK_KEY=sk-...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.0-flash-001
```

---

## CLI Usage

### Standard Interface
```bash
python -m bodhan_dub --input path/to/video.mp4 --langs hi,ta,bn --out ./out
```

### Advanced Options
```bash
python -m bodhan_dub \
  --input path/to/video.mp4 \
  --langs hi,te,mr \
  --out ./out \
  --glossary glossary.example.yaml \
  --voices voices.example.yaml \
  --code-mixed \
  --max-cost-inr 50.0 \
  --concurrency 4
```

### Cost Estimation Mode
```bash
python -m bodhan_dub --input path/to/video.mp4 --langs hi,ta,bn --estimate-only
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input` | Path | *Required* | Path to source video file (MP4, MOV, MKV, WebM, 1–15 min) |
| `--langs` | String | *Required* | Comma-separated target Indic language codes (e.g. `hi,ta,bn`) |
| `--out` | Path | `./out` | Output destination directory |
| `--glossary` | Path | `None` | Path to custom terminology glossary YAML |
| `--voices` | Path | `None` | Path to custom voice configuration YAML |
| `--code-mixed` | Flag | `False` | Retain English technical vocabulary in Indic translation |
| `--max-cost-inr` | Float | `None` | Budget limit in INR; pipeline stops cleanly if reached |
| `--estimate-only` | Flag | `False` | Print cost estimate table in INR and exit |
| `--concurrency` | Int | `4` | Maximum parallel language worker concurrency |
| `--no-cache` | Flag | `False` | Disable content-addressed sha256 disk cache |
| `--resume` | Flag | `False` | Resume execution from existing run checkpoints |
| `--keep-work` | Flag | `False` | Retain intermediate extracted audio and stems |
| `--separate-music` | Flag | `False` | Separate background music using Demucs |
| `--speaker-voices` | Choice | `single` | Voice strategy: `single` (default) or `auto` (pitch heuristic) |
| `--log-level` | Choice | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Supported Production Languages

Loaded dynamically from `languages.yaml`:
- **Assamese (`as`)**
- **Bengali (`bn`)**
- **Gujarati (`gu`)**
- **Hindi (`hi`)**
- **Kannada (`kn`)**
- **Malayalam (`ml`)**
- **Marathi (`mr`)**
- **Odia (`or`)**
- **Punjabi (`pa`)**
- **Tamil (`ta`)**
- **Telugu (`te`)**
- **Urdu (`ur`)**

---

## Output Directory Structure

```
out/
  report.json                 # Comprehensive run report (costs, sync, stretch, CER, targets)
  transcript.en.json          # Formatted English sentence segments with start/end
  run.log.jsonl               # Structured JSONL log events with redacted keys
  manifest.json               # Stage and segment checkpoints
  <lang>/
    video.<lang>.mp4          # AI-dubbed video in target language (AAC 192k, 48kHz)
    subtitles.<lang>.srt      # Grapheme-wrapped SRT subtitles
    subtitles.<lang>.vtt      # WebVTT subtitles
    segments.<lang>.json      # Source text, translation, audio duration, and QA metrics
```

---

## Exit Codes

- `0`: All requested languages processed successfully (status `ok` or `degraded`).
- `1`: At least one language failed (partial outputs and `report.json` preserved).
- `2`: Preflight or usage error (missing streams, duration outside bounds, unknown language, budget exceeded).
- `130`: Interrupted by user (SIGINT / Ctrl-C; writes partial manifest).

---

## Development & Testing

```bash
make setup     # Install dependencies and preflight environment
make check     # Run ruff linter and full pytest test suite (77 tests)
make smoke     # Run offline end-to-end smoke test on synthetic video fixture
make estimate  # Run cost estimate on demo media
```

---

## AI Dubbing Disclosure

All video outputs produced by this software are marked with container-level metadata:
- **MP4 Comment:** `comment="AI-dubbed with Bodhan Indic-Speak"`
- **Audio Stream Tag:** `language=<lang>`
- **Report Label:** `ai_label: "AI-dubbed with Bodhan Indic-Speak"`
