"""
Content-addressed disk cache for Bodhan API responses.
Key = sha256(service, model, sorted(params), input bytes/text, cache_version).
Cache hits cost ₹0 and are tracked in the cost ledger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class DiskCache:
    """Stores API responses keyed by deterministic sha256 hash."""

    def __init__(self, cache_dir: Path, version: str = "v1", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(
        self,
        service: str,
        model: str,
        params: Dict[str, Any],
        payload: Union[str, bytes],
    ) -> str:
        """Computes deterministic sha256 hash for a request."""
        h = hashlib.sha256()
        h.update(service.encode("utf-8"))
        h.update(b"|")
        h.update(model.encode("utf-8"))
        h.update(b"|")
        # Canonical JSON string of sorted params
        norm_params = json.dumps(params, sort_keys=True)
        h.update(norm_params.encode("utf-8"))
        h.update(b"|")
        if isinstance(payload, bytes):
            h.update(payload)
        else:
            h.update(payload.encode("utf-8"))
        h.update(b"|")
        h.update(self.version.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[Tuple[Any, float]]:
        """
        Looks up cached entry.
        Returns (data, original_cost) if found, else None.
        """
        if not self.enabled:
            return None

        meta_path = self.cache_dir / f"{key}.meta.json"
        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data_type = meta.get("data_type", "json")
            cost = float(meta.get("cost", 0.0))

            if data_type == "binary":
                bin_path = self.cache_dir / f"{key}.bin"
                if bin_path.exists():
                    return bin_path.read_bytes(), cost
            elif data_type == "json":
                json_path = self.cache_dir / f"{key}.json"
                if json_path.exists():
                    return json.loads(json_path.read_text(encoding="utf-8")), cost
            elif data_type == "text":
                txt_path = self.cache_dir / f"{key}.txt"
                if txt_path.exists():
                    return txt_path.read_text(encoding="utf-8"), cost
        except Exception:
            return None

        return None

    def put(
        self,
        key: str,
        data: Union[Dict[str, Any], list, str, bytes],
        cost: float,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stores entry and metadata in cache."""
        if not self.enabled:
            return

        meta = {
            "key": key,
            "cost": cost,
            "extra": extra_meta or {},
        }

        try:
            if isinstance(data, bytes):
                meta["data_type"] = "binary"
                (self.cache_dir / f"{key}.bin").write_bytes(data)
            elif isinstance(data, (dict, list)):
                meta["data_type"] = "json"
                (self.cache_dir / f"{key}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                meta["data_type"] = "text"
                (self.cache_dir / f"{key}.txt").write_text(str(data), encoding="utf-8")

            (self.cache_dir / f"{key}.meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception:
            pass  # Non-fatal if cache write fails
