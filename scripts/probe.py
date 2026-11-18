#!/usr/bin/env python3
"""
Phase 0 API Probe — Verify Bodhan API facts with minimal spend (≤ ₹1).

Tests:
  1. Transcribe a ~10s synthetic clip (timestamps returned?)
  2. Translate 3 English sentences (incl. one code-mixed target)
  3. Synthesize 2 short sentences per available voice (chars/sec, max length)

Saves sanitised sample responses under tests/fixtures/.
"""
from __future__ import annotations

import io
import json
import os
import struct
import time
import wave
from pathlib import Path

import httpx
import soundfile as sf

# ---------------------------------------------------------------------------
# Load .env from workspace root (two levels up from scripts/)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

# Try multiple .env locations
for env_path in [
    Path(__file__).resolve().parent.parent.parent.parent / ".env",  # workspace root
    Path(__file__).resolve().parent.parent / ".env",                # project root
]:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[probe] Loaded .env from {env_path}")
        break
else:
    print("[probe] WARNING: No .env file found, relying on environment variables")

# ---------------------------------------------------------------------------
# Config — read keys, NEVER print them
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("BODHAN_BASE_URL", "https://api.bodhan.ai/v1")
TRANSCRIBE_KEY = os.getenv("BODHAN_TRANSCRIBE_KEY", "")
TRANSLATE_KEY = os.getenv("BODHAN_TRANSLATE_KEY", "")
SPEAK_KEY = os.getenv("BODHAN_SPEAK_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

results: dict = {"probe_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "facts": {}}

def redact(s: str) -> str:
    if len(s) <= 8:
        return "***"
    return s[:4] + "..." + s[-4:]

def save_fixture(name: str, data):
    path = FIXTURE_DIR / name
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [saved] {path.name}")


# ---------------------------------------------------------------------------
# Generate a synthetic ~10s WAV with a 440 Hz tone (pure sine wave)
# ---------------------------------------------------------------------------
def make_test_wav(duration_s: float = 10.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple sine-wave WAV file in memory."""
    import math
    n_samples = int(duration_s * sample_rate)
    freq = 440.0
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
        samples.append(val)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. TRANSCRIBE PROBE
# ---------------------------------------------------------------------------
def probe_transcribe():
    print("\n=== PROBE 1: Indic-Transcribe ===")
    if not TRANSCRIBE_KEY:
        print("  SKIP: BODHAN_TRANSCRIBE_KEY not set")
        results["facts"]["transcribe"] = {"status": "skipped", "reason": "no key"}
        return

    wav_data = make_test_wav(10.0)
    print(f"  Generated {len(wav_data)} byte test WAV (10s, 16kHz, mono)")

    client = httpx.Client(timeout=30.0)
    headers = {"Authorization": f"Bearer {TRANSCRIBE_KEY}"}

    # Test 1a: Default response format
    print("  [1a] Default transcription...")
    try:
        resp = client.post(
            f"{BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"model": "indic-transcribe", "language": "en"},
        )
        result_1a = {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.json() if resp.status_code == 200 else resp.text[:500],
        }
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            body = resp.json()
            print(f"  Response keys: {list(body.keys())}")
            print(f"  Has 'text': {'text' in body}")
            print(f"  Has 'segments': {'segments' in body}")
            print(f"  Has 'words': {'words' in body}")
            print(f"  Text preview: {str(body.get('text', ''))[:100]}")
        else:
            print(f"  Error: {resp.text[:200]}")
    except Exception as e:
        result_1a = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Test 1b: Try verbose_json
    print("  [1b] Trying response_format=verbose_json...")
    try:
        resp = client.post(
            f"{BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"model": "indic-transcribe", "language": "en",
                  "response_format": "verbose_json"},
        )
        result_1b = {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
        }
        if resp.status_code == 200:
            body = resp.json()
            result_1b["body_keys"] = list(body.keys())
            result_1b["has_segments"] = "segments" in body
            result_1b["has_words"] = "words" in body
            if "segments" in body and body["segments"]:
                result_1b["sample_segment"] = body["segments"][0]
                result_1b["segment_count"] = len(body["segments"])
            print(f"  Status: {resp.status_code}")
            print(f"  Response keys: {list(body.keys())}")
            print(f"  Has 'segments': {'segments' in body}")
            if "segments" in body and body["segments"]:
                print(f"  First segment: {json.dumps(body['segments'][0], ensure_ascii=False)[:200]}")
        else:
            result_1b["error"] = resp.text[:300]
            print(f"  Status: {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        result_1b = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Test 1c: Try indic-transcribe-flex
    print("  [1c] Trying model=indic-transcribe-flex...")
    try:
        resp = client.post(
            f"{BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"model": "indic-transcribe-flex", "language": "en"},
        )
        result_1c = {"status_code": resp.status_code}
        if resp.status_code == 200:
            result_1c["accepted"] = True
            print("  ACCEPTED (status 200)")
        else:
            result_1c["accepted"] = False
            result_1c["error"] = resp.text[:200]
            print(f"  REJECTED: {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        result_1c = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Collect rate limit headers
    rate_headers = {}
    for h in ["x-ratelimit-limit-requests", "x-ratelimit-limit-parallel-requests",
              "x-ratelimit-remaining-requests"]:
        if h in result_1a.get("headers", {}):
            rate_headers[h] = result_1a["headers"][h]

    results["facts"]["transcribe"] = {
        "default_response": {k: v for k, v in result_1a.items() if k != "headers"},
        "verbose_json": result_1b,
        "flex_model": result_1c,
        "rate_limit_headers": rate_headers,
    }

    # Save sanitised fixture (strip auth headers)
    fixture = {k: v for k, v in result_1a.items() if k != "headers"}
    save_fixture("transcribe_default_response.json", fixture)
    if "body_keys" in result_1b:
        save_fixture("transcribe_verbose_response.json", {k: v for k, v in result_1b.items() if k != "headers"})


# ---------------------------------------------------------------------------
# 2. TRANSLATE PROBE
# ---------------------------------------------------------------------------
def probe_translate():
    print("\n=== PROBE 2: Indic-Translate ===")
    if not TRANSLATE_KEY:
        print("  SKIP: BODHAN_TRANSLATE_KEY not set")
        results["facts"]["translate"] = {"status": "skipped", "reason": "no key"}
        return

    client = httpx.Client(timeout=30.0)
    headers = {
        "Authorization": f"Bearer {TRANSLATE_KEY}",
        "Content-Type": "application/json",
    }

    test_cases = [
        {
            "name": "simple_hi",
            "text": "The class starts at nine o'clock in the morning.",
            "source": "en", "target": "hi",
        },
        {
            "name": "technical_ta",
            "text": "The speed of light is approximately 3 times 10 to the power 8 meters per second.",
            "source": "en", "target": "ta",
        },
        {
            "name": "code_mixed_hi",
            "text": "Please submit your assignment by Friday evening.",
            "source": "en", "target": "Hindi (code-mixed)",
        },
    ]

    translate_results = {}
    for tc in test_cases:
        print(f"  [{tc['name']}] en → {tc['target']}...")
        payload = {
            "model": "indic-translate",
            "messages": [{"role": "user", "content": tc["text"]}],
            "source_language_code": tc["source"],
            "target_language_code": tc["target"],
        }
        try:
            resp = client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                body = resp.json()
                translated = body["choices"][0]["message"]["content"]
                usage = body.get("usage", {})
                print("    Status: 200")
                print(f"    Translation: {translated[:100]}")
                print(f"    Usage: {usage}")
                translate_results[tc["name"]] = {
                    "status": 200,
                    "translation_preview": translated[:200],
                    "usage": usage,
                    "response_keys": list(body.keys()),
                    "choice_keys": list(body["choices"][0].keys()) if body.get("choices") else [],
                }
            else:
                print(f"    Status: {resp.status_code} — {resp.text[:200]}")
                translate_results[tc["name"]] = {
                    "status": resp.status_code,
                    "error": resp.text[:300],
                }
        except Exception as e:
            translate_results[tc["name"]] = {"error": str(e)}
            print(f"    ERROR: {e}")

    # Test back-translation (target → en)
    print("  [back_translate] Translate back to English without source_language_code...")
    if "simple_hi" in translate_results and translate_results["simple_hi"].get("status") == 200:
        hi_text = translate_results["simple_hi"]["translation_preview"]
        payload = {
            "model": "indic-translate",
            "messages": [{"role": "user", "content": hi_text}],
            "target_language_code": "en",
        }
        try:
            resp = client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                body = resp.json()
                back = body["choices"][0]["message"]["content"]
                print(f"    Back-translation: {back[:100]}")
                translate_results["back_translate"] = {
                    "status": 200,
                    "text_preview": back[:200],
                    "source_omitted": True,
                }
            else:
                translate_results["back_translate"] = {"status": resp.status_code, "error": resp.text[:200]}
                print(f"    Status: {resp.status_code}")
        except Exception as e:
            translate_results["back_translate"] = {"error": str(e)}

    results["facts"]["translate"] = translate_results
    save_fixture("translate_responses.json", translate_results)


# ---------------------------------------------------------------------------
# 3. SPEAK PROBE
# ---------------------------------------------------------------------------
def probe_speak():
    print("\n=== PROBE 3: Indic-Speak ===")
    if not SPEAK_KEY:
        print("  SKIP: BODHAN_SPEAK_KEY not set")
        results["facts"]["speak"] = {"status": "skipped", "reason": "no key"}
        return

    client = httpx.Client(timeout=60.0)
    headers = {
        "Authorization": f"Bearer {SPEAK_KEY}",
        "Content-Type": "application/json",
    }

    # Known voices to test
    test_voices = [
        {"voice": "Kavya", "lang": "hi", "text": "नमस्ते, आज का मौसम बहुत अच्छा है।"},
        {"voice": "Amit", "lang": "hi", "text": "कक्षा सुबह नौ बजे शुरू होती है।"},
        {"voice": "Ishita", "lang": "bn", "text": "আজ আবহাওয়া খুব ভালো।"},
        {"voice": "Sourav", "lang": "bn", "text": "ক্লাস সকাল নটায় শুরু হয়।"},
    ]

    speak_results = {}
    for tv in test_voices:
        name = f"{tv['voice']}_{tv['lang']}"
        print(f"  [{name}] Synthesizing...")
        payload = {
            "model": "indic-speak",
            "input": tv["text"],
            "voice": tv["voice"],
            "instructions": json.dumps({"lang": tv["lang"]}),
        }
        try:
            resp = client.post(f"{BASE_URL}/audio/speech", headers=headers, json=payload)
            rate_headers = {h: resp.headers.get(h) for h in [
                "x-ratelimit-limit-requests",
                "x-ratelimit-limit-parallel-requests",
                "x-ratelimit-remaining-requests",
            ] if resp.headers.get(h)}

            if resp.status_code == 200:
                audio_bytes = resp.content
                content_type = resp.headers.get("content-type", "unknown")
                print(f"    Status: 200, Content-Type: {content_type}, Size: {len(audio_bytes)} bytes")

                # Analyze WAV
                try:
                    buf = io.BytesIO(audio_bytes)
                    info = sf.info(buf)
                    duration = info.duration
                    sample_rate = info.samplerate
                    channels = info.channels
                    char_count = len(tv["text"])
                    cps = char_count / duration if duration > 0 else 0
                    print(f"    Sample rate: {sample_rate} Hz, Channels: {channels}, Duration: {duration:.2f}s")
                    print(f"    Characters: {char_count}, CPS: {cps:.1f}")
                    speak_results[name] = {
                        "status": 200,
                        "content_type": content_type,
                        "audio_bytes": len(audio_bytes),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "duration_s": round(duration, 3),
                        "char_count": char_count,
                        "chars_per_second": round(cps, 2),
                        "rate_headers": rate_headers,
                    }
                    # Save first WAV as fixture (small)
                    if len(speak_results) <= 1:
                        save_fixture(f"tts_sample_{name}.wav", audio_bytes)
                except Exception as e2:
                    print(f"    WAV parse error: {e2}")
                    speak_results[name] = {"status": 200, "wav_parse_error": str(e2),
                                           "audio_bytes": len(audio_bytes), "rate_headers": rate_headers}
            else:
                error_text = resp.text[:300]
                print(f"    Status: {resp.status_code} — {error_text}")
                speak_results[name] = {"status": resp.status_code, "error": error_text, "rate_headers": rate_headers}
        except Exception as e:
            speak_results[name] = {"error": str(e)}
            print(f"    ERROR: {e}")

    # Test TTS with longer text to find character limit
    print("\n  [char_limit] Testing increasing text lengths...")
    test_text_base = "यह एक परीक्षण वाक्य है। "  # ~24 chars
    for multiplier in [5, 10, 20, 40]:
        long_text = test_text_base * multiplier
        char_count = len(long_text)
        print(f"    Testing {char_count} characters...")
        payload = {
            "model": "indic-speak",
            "input": long_text,
            "voice": "Kavya",
            "instructions": json.dumps({"lang": "hi"}),
        }
        try:
            resp = client.post(f"{BASE_URL}/audio/speech", headers=headers, json=payload)
            if resp.status_code == 200:
                buf = io.BytesIO(resp.content)
                info = sf.info(buf)
                print(f"      OK: {char_count} chars → {info.duration:.2f}s")
                speak_results[f"length_{char_count}"] = {
                    "chars": char_count, "status": 200, "duration_s": round(info.duration, 3)
                }
            else:
                print(f"      REJECTED at {char_count} chars: {resp.status_code}")
                speak_results[f"length_{char_count}"] = {
                    "chars": char_count, "status": resp.status_code, "error": resp.text[:200]
                }
                break  # found the limit
        except Exception as e:
            speak_results[f"length_{char_count}"] = {"chars": char_count, "error": str(e)}
            break

    results["facts"]["speak"] = speak_results
    save_fixture("speak_responses.json", speak_results)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BODHAN API PROBE — Phase 0 Recon")
    print(f"Base URL: {BASE_URL}")
    print(f"Keys: transcribe={redact(TRANSCRIBE_KEY)}, translate={redact(TRANSLATE_KEY)}, speak={redact(SPEAK_KEY)}")
    print("=" * 60)

    missing = []
    if not TRANSCRIBE_KEY:
        missing.append("BODHAN_TRANSCRIBE_KEY")
    if not TRANSLATE_KEY:
        missing.append("BODHAN_TRANSLATE_KEY")
    if not SPEAK_KEY:
        missing.append("BODHAN_SPEAK_KEY")
    if missing:
        print(f"\nWARNING: Missing keys: {', '.join(missing)}")
        print("Will skip probes for missing keys.\n")

    probe_transcribe()
    probe_translate()
    probe_speak()

    # Save combined results
    save_fixture("probe_results.json", results)

    print("\n" + "=" * 60)
    print("PROBE COMPLETE")
    print(f"Results saved to {FIXTURE_DIR / 'probe_results.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
