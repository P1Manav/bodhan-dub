# bodhan_dub: Autonomous English-to-Indic Video Dubbing Pipeline

An autonomous, production-grade video dubbing system that translates and dubs English videos into **13 production-tier Indian languages** using **strictly Bodhan Indic AI models** (`indic-transcribe`, `indic-translate`, `indic-speak`).

All generated output media is formally labeled as **AI-dubbed** via container metadata tags, structured pipeline reports (`report.json`), and subtitle sidecars.

---

## Architecture Overview

The system strictly adheres to the five architectural workflows specified in `docs/`:

1. **System Architecture (`docs/ARCHITECTURE.md` Diagram 1):** Preflight validation, Silero ONNX VAD + ffmpeg fallback, FastConformer 29s chunk planner, Indic-Transcribe ASR, sentence slot segmentation with slack calculations.
2. **End-to-End Decision Pipeline (Diagram 2):** Isolated per-language worker pipelines with concurrency semaphores, circuit breakers, and content-addressed cache.
3. **Per-Segment Duration Fit & QA Loop (Diagram 3):** Indic-Speak synthesis, acoustic QA (CER, repetition, tail omission check via back-transcription), bounded 3-attempt retry ladder, and pitch-preserving time stretch (`atempo` capped at 1.20x).
4. **Segment Status Lifecycle (Diagram 4):** Deterministic state machine governing slot transitions (`raw` -> `transcribed` -> `translated` -> `synthesized` -> `qa_passed`/`degraded` -> `fit_ok`/`borrowed_slack`/`overrun`).
5. **Timeline Synchronization & Audio Mixing (Diagram 5):** Drift-bounded speech placement, 40ms dialogue ducking with ambient preservation, EBU R128 loudness normalization (-16.0 LUFS ± 1.0 LUFS, -1.0 dBTP true-peak ceiling), and MP4 muxing with dual subtitles (SRT + VTT).

---

## Hard Engineering Constraints

- **Indic-Only Speech & Translation:** All speech recognition goes through `indic-transcribe`, translation through `indic-translate`, and synthesis through `indic-speak`. No Whisper, no IndicTrans2, no commercial voice/cloud speech APIs.
- **Strict English-Only LLM:** The LLM never writes, rewrites, or translates Indian-language text. A character-level Unicode policy guard rejects any output containing Indic or Arabic characters (`U+0600–U+06FF`, `U+0900–U+0DFF`, `U+1C50–U+1C7F`).
- **OpenAI-Compatible LLM Client:** Schema-validated JSON-in-text queries with 1 automatic repair retry and deterministic fallbacks.
- **No Cloned Voices:** Uses calibrated Bodhan production voice profiles.
- **Zero Secrets / Media in Repository:** API keys are read strictly from environment variables. `.env` is ignored by git; `.env.example` is provided as reference.

---

## Repository Structure

```
.
├── pyproject.toml             # Pinned Python 3.11+ dependencies
├── Makefile                   # setup, check, smoke, estimate targets
├── Dockerfile                 # Containerized runtime with ffmpeg
├── languages.yaml             # Registry of 13 production languages & voices
├── glossary.example.yaml      # Terminology mapping template
├── voices.example.yaml        # Voice override template
├── README.md                  # Project root documentation
├── .gitignore                 # Repository-level ignore rules
├── .env.example               # Sanitized configuration template
├── docs/                      # Architectural specifications & decision records
│   ├── ARCHITECTURE.md        # 5 Mermaid architecture diagrams
│   ├── API_FACTS.md           # Live Bodhan API probe benchmarks & limits
│   ├── DECISIONS.md           # Architecture Decision Records (ADRs)
│   ├── DESIGN_NOTES.md        # Detailed module design specifications
│   ├── FAILURE_MODES.md       # Edge cases and mitigation strategies
│   └── PR_DESCRIPTION.md     # Production pull request documentation
├── scripts/
│   └── probe.py               # Live API verification probe
├── src/bodhan_dub/            # Pipeline implementation
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Settings & environment parser
│   ├── languages.py           # Language registry loader
│   ├── preflight.py           # ffprobe, disk, budget preflight checks
│   ├── orchestrator.py        # Isolated multi-language pipeline runner
│   ├── policy.py              # Security guards & Unicode checkers
│   ├── logging.py             # Structured JSONL pipeline logger
│   ├── manifest.py            # Checkpoint & resume manifest
│   ├── gateway/               # Bodhan client, LLM client, cache, budget guard
│   ├── media/                 # VAD, audio extraction, loudness, mixer, muxer
│   ├── stages/                # ASR, translation, glossary, TTS, fit, QA, scheduler, subtitles, report
│   └── text/                  # Unicode grapheme clusters, CER, spans, normalization
└── tests/                     # 80 unit, e2e smoke, and mock tests (100% pass)
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- `ffmpeg` (version 7.0+) and `ffprobe` on system PATH

### Installation
```bash
# Create virtual environment and install exact pinned dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -e .

# Or using Makefile
make setup
```

### Environment Configuration
Copy `.env.example` to `.env` and configure your API credentials:
```bash
cp .env.example .env
```
Required environment variables:
```env
BODHAN_TRANSCRIBE_KEY=sk-...
BODHAN_TRANSLATE_KEY=sk-...
BODHAN_SPEAK_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
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

## Running Tests and Linting

```bash
# Run all 80 unit, e2e smoke, and mock tests
pytest tests/ -v

# Run linter
ruff check .

# Or using Makefile
make check
```

---

## Exit Codes

- `0`: All requested languages processed successfully (status `ok` or `degraded`).
- `1`: At least one language failed (partial outputs and `report.json` preserved).
- `2`: Preflight or usage error (missing streams, duration outside bounds, unknown language, budget exceeded).
- `130`: Interrupted by user (SIGINT / Ctrl-C; writes partial manifest).
