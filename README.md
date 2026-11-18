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
├── README.md                      # Project root documentation
├── .gitignore                     # Repository-level ignore rules
├── .env.example                   # Sanitized configuration template
├── examples/video-dubbing/        # Core pipeline package
│   ├── pyproject.toml             # Pinned Python 3.11+ dependencies
│   ├── Makefile                   # setup, check, smoke, estimate targets
│   ├── Dockerfile                 # Containerized runtime with ffmpeg
│   ├── languages.yaml             # Registry of 13 production languages & voices
│   ├── glossary.example.yaml      # Terminology mapping template
│   ├── voices.example.yaml        # Voice override template
│   ├── docs/                      # Architectural specifications & decision records
│   │   ├── ARCHITECTURE.md        # 5 Mermaid architecture diagrams
│   │   ├── API_FACTS.md           # Live Bodhan API probe benchmarks & limits
│   │   ├── DECISIONS.md           # Architecture Decision Records (ADRs)
│   │   ├── DESIGN_NOTES.md        # Detailed module design specifications
│   │   ├── FAILURE_MODES.md       # Edge cases and mitigation strategies
│   │   └── PR_DESCRIPTION.md     # Production pull request documentation
│   ├── scripts/
│   │   └── probe.py               # Live API verification probe
│   ├── src/bodhan_dub/            # Pipeline implementation
│   │   ├── cli.py                 # Command-line interface
│   │   ├── config.py              # Settings & environment parser
│   │   ├── languages.py           # Language registry loader
│   │   ├── preflight.py           # ffprobe, disk, budget preflight checks
│   │   ├── orchestrator.py        # Isolated multi-language pipeline runner
│   │   ├── policy.py              # Security guards & Unicode checkers
│   │   ├── logging.py             # Structured JSONL pipeline logger
│   │   ├── manifest.py            # Checkpoint & resume manifest
│   │   ├── gateway/               # Bodhan client, LLM client, cache, budget guard
│   │   ├── media/                 # VAD, audio extraction, loudness, mixer, muxer
│   │   ├── stages/                # ASR, translation, glossary, TTS, fit, QA, scheduler, subtitles, report
│   │   └── text/                  # Unicode grapheme clusters, CER, spans, normalization
│   └── tests/                     # 80 unit, e2e smoke, and mock tests (100% pass)
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- `ffmpeg` (version 7.0+) and `ffprobe` on system PATH

### Installation
```bash
cd examples/video-dubbing
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

### Running the Dubbing Pipeline
```bash
python -m bodhan_dub --input path/to/video.mp4 --langs hi,ta,bn --out ./out
```

### Running Tests and Linting
```bash
cd examples/video-dubbing
pytest tests/ -v
ruff check .
```
