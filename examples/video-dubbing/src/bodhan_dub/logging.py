"""
Structured JSONL logging for bodhan_dub.
Ensures every log entry includes run_id, stage, lang, segment_id, and attempt,
and automatically redacts sensitive tokens/keys.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Regex pattern to detect sensitive api-key style strings for redaction
SECRET_PATTERN = re.compile(r"sk-[a-zA-Z0-9_-]{16,}")


def sanitize_text(text: str) -> str:
    """Replaces API keys in log strings with safe redacted placeholders."""
    if not isinstance(text, str):
        return text
    return SECRET_PATTERN.sub(lambda m: f"{m.group(0)[:6]}...{m.group(0)[-4:]}", text)


class JsonlHandler(logging.Handler):
    """Appends structured JSON log records to a .jsonl file."""

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = self.format_entry(record)
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:
            self.handleError(record)

    def format_entry(self, record: logging.LogRecord) -> Dict[str, Any]:
        msg = record.getMessage()
        sanitized_msg = sanitize_text(msg)

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
            "run_id": getattr(record, "run_id", "default"),
            "stage": getattr(record, "stage", "system"),
            "lang": getattr(record, "lang", None),
            "segment_id": getattr(record, "segment_id", None),
            "attempt": getattr(record, "attempt", 1),
        }
        # Include any extra metadata passed
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            entry["extra"] = record.extra_data
        return entry

    def close(self) -> None:
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()
        super().close()


class PipelineLogger:
    """Wrapper that injects pipeline context (run_id, stage, lang, etc.) into logs."""

    def __init__(
        self,
        name: str = "bodhan_dub",
        run_id: str = "run-001",
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
    ):
        self.run_id = run_id
        self.logger = logging.getLogger(f"{name}.{run_id}")
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler with clean human-readable output
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s [%(stage)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        # JSONL file handler if path provided
        if log_file:
            jh = JsonlHandler(log_file)
            jh.setLevel(level)
            self.logger.addHandler(jh)

    def log(
        self,
        level: int,
        msg: str,
        stage: str = "system",
        lang: Optional[str] = None,
        segment_id: Optional[str] = None,
        attempt: int = 1,
        **extra: Any,
    ) -> None:
        self.logger.log(
            level,
            msg,
            extra={
                "run_id": self.run_id,
                "stage": stage,
                "lang": lang,
                "segment_id": segment_id,
                "attempt": attempt,
                "extra_data": extra if extra else None,
            },
        )

    def info(self, msg: str, **kwargs: Any) -> None:
        self.log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self.log(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self.log(logging.DEBUG, msg, **kwargs)

    def flush(self) -> None:
        """Flushes all log handlers."""
        for h in self.logger.handlers:
            h.flush()

    def save_log(self, filepath: Optional[Path] = None) -> None:
        """Flushes active handlers to disk."""
        self.flush()
