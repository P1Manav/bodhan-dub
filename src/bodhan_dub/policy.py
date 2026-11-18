"""
Policy Guard: Structural enforcement of platform constraints.
1. Host Allowlist: Rejects any HTTP traffic outside Bodhan, LLM host, and localhost/tests.
2. English-only LLM Guard: Rejects prompts/outputs containing Indic or Arabic script characters.
3. Forbidden Dependency Audit: Ensures no disallowed speech/translate libraries are present.
"""

from __future__ import annotations

from typing import List, Tuple
from urllib.parse import urlparse

# Unicode ranges for Indic and Arabic-derived scripts
# U+0600-U+06FF: Arabic (Urdu, Sindhi, Kashmiri)
# U+08A0-U+08FF: Arabic Extended-A
# U+0900-U+0DFF: Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Sinhala)
# U+1C50-U+1C7F: Ol Chiki (Santali)
# U+ABC0-U+ABFF: Meetei Mayek (Manipuri)
INDIC_AND_ARABIC_RANGES: List[Tuple[int, int]] = [
    (0x0600, 0x06FF),
    (0x08A0, 0x08FF),
    (0x0900, 0x0DFF),
    (0x1C50, 0x1C7F),
    (0xABC0, 0xABFF),
]

FORBIDDEN_DEPENDENCIES = [
    "whisper",
    "faster-whisper",
    "faster_whisper",
    "vosk",
    "pytesseract",
    "googletrans",
    "deep-translator",
    "deep_translator",
    "indictrans",
    "indictrans2",
    "gtts",
    "pyttsx3",
    "edge-tts",
    "edge_tts",
    "coqui",
    "elevenlabs",
    "azure-cognitiveservices-speech",
    "azure_cognitiveservices_speech",
    "google-cloud-speech",
    "google_cloud_speech",
    "google-cloud-translate",
    "google_cloud_translate",
    "google-cloud-texttospeech",
    "google_cloud_texttospeech",
    "speechrecognition",
]


class PolicyViolation(Exception):
    """Raised when an action violates the core pipeline constraints."""

    pass


def contains_indic_or_arabic(text: str) -> bool:
    """Returns True if any character in text falls into Indic or Arabic script Unicode blocks."""
    if not text:
        return False
    for char in text:
        cp = ord(char)
        for start, end in INDIC_AND_ARABIC_RANGES:
            if start <= cp <= end:
                return True
    return False


def validate_llm_text(text: str, context: str = "LLM input/output") -> str:
    """
    Validates that text sent to or received from the LLM contains NO Indic or Arabic characters.
    Raises PolicyViolation if any disallowed character is found.
    """
    if contains_indic_or_arabic(text):
        raise PolicyViolation(
            f"Policy Guard Violation: {context} contains non-English / Indic-script characters! "
            "LLMs may strictly process and generate English text only."
        )
    return text


guard_llm_output_english_only = validate_llm_text


class HostAllowlistGuard:
    """HTTP client event hook enforcing that requests only target allowlisted hosts."""

    def __init__(self, allowed_hosts: List[str]):
        self.allowed_hosts = set(h.lower() for h in allowed_hosts)
        # Always allow test / local hosts
        self.allowed_hosts.add("localhost")
        self.allowed_hosts.add("127.0.0.1")
        self.allowed_hosts.add("testserver")

    def check_url(self, url: str) -> None:
        parsed = urlparse(str(url))
        host = (parsed.hostname or "").lower()
        if host not in self.allowed_hosts:
            raise PolicyViolation(
                f"Policy Guard Violation: Outgoing HTTP request to forbidden host '{host}'! "
                f"Allowed hosts: {sorted(list(self.allowed_hosts))}"
            )

    async def async_request_hook(self, request) -> None:
        """httpx async event hook."""
        self.check_url(str(request.url))

    def sync_request_hook(self, request) -> None:
        """httpx sync event hook."""
        self.check_url(str(request.url))
