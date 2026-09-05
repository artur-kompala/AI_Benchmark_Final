"""Inicjalizacja klienta Supabase (TYLKO odczyt) z danych logowania w .env.

Dashboard jest niezalezna warstwa od benchmark_runner (patrz docs/architecture.md) -
ten plik jest swiadomie osobna kopia analogicznego modulu w benchmark_runner, a nie
importem wspoldzielonego kodu.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

_client: Client | None = None

# .../dashboard/.env - jawna sciezka zamiast polegania na cwd (load_dotenv() bez
# argumentu szuka .env wzgledem katalogu roboczego procesu, ktory przy uruchomieniu
# przez `streamlit run` moze byc inny niz katalog projektu dashboard/).
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
            "do .env i uzupelnij dane logowania (patrz ../docs/setup_supabase.md)."
        )

    _client = create_client(url, key)
    return _client
