"""Inicjalizacja klienta Supabase z danych logowania w .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

_client: Client | None = None

# .../benchmark_runner/.env - jawna sciezka zamiast polegania na cwd (load_dotenv()
# bez argumentu szuka .env wzgledem katalogu roboczego procesu).
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class SupabaseConfigError(Exception):
    pass


def get_supabase_client(force_reload: bool = False) -> Client:
    global _client
    if _client is not None and not force_reload:
        return _client

    load_dotenv(dotenv_path=_ENV_PATH)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SupabaseConfigError(
            "Brak SUPABASE_URL/SUPABASE_KEY w zmiennych srodowiskowych. Skopiuj .env.example "
            "do .env i uzupelnij dane logowania (patrz docs/setup_supabase.md)."
        )

    _client = create_client(url, key)
    return _client
