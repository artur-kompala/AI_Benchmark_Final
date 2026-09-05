"""Hashowanie konfiguracji do idempotencji/deduplikacji przebiegow (runs.config_hash)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_config(config_snapshot: dict[str, Any]) -> str:
    """Deterministyczny sha256 ze slownika konfiguracji (klucze posortowane,
    zeby ta sama konfiguracja w innej kolejnosci pol dawala ten sam hash)."""
    canonical = json.dumps(config_snapshot, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
