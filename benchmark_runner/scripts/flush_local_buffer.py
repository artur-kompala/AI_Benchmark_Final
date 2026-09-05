"""Ponownie wysyla do Supabase wyniki zbuforowane lokalnie w .local_buffer/pending_runs.jsonl
(Blad 2 - zapisy do Supabase odrzucane, np. przez check constraint na devices.device_type,
lub przerwane polaczenie siecowe - patrz storage/local_buffer.py).

Uruchamiaj DOPIERO po naprawieniu przyczyny zrodlowej bledu zapisu (np. po uruchomieniu
db/migrations/0006_intel_gpu.sql w Supabase SQL Editor, patrz docs/setup_supabase.md) -
w przeciwnym razie kazdy zbuforowany pakiet po prostu wroci do bufora z tym samym bledem.

Uzycie:
    python scripts/flush_local_buffer.py [--local-buffer-dir ./.local_buffer/]

Wymaga aktywnego venv i uzupelnionego .env (SUPABASE_URL/SUPABASE_KEY, patrz
docs/setup_supabase.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_runner.storage.local_buffer import LocalBuffer  # noqa: E402
from benchmark_runner.storage.repository import RunRepository  # noqa: E402
from benchmark_runner.storage.supabase_client import SupabaseConfigError, get_supabase_client  # noqa: E402
from benchmark_runner.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flush_local_buffer",
        description="Ponownie wysyla zbuforowane lokalnie wyniki (.local_buffer/pending_runs.jsonl) do Supabase",
    )
    parser.add_argument(
        "--local-buffer-dir",
        default="./.local_buffer/",
        help="Katalog z lokalnym buforem wynikow (domyslnie ./.local_buffer/, jak w cli.py --local-buffer-dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_arg_parser().parse_args(argv)

    try:
        client = get_supabase_client()
    except SupabaseConfigError as exc:
        logger.error(str(exc))
        return 1

    repository = RunRepository(client)
    local_buffer = LocalBuffer(buffer_dir=args.local_buffer_dir)

    pending_before = local_buffer.pending_count()
    if pending_before == 0:
        logger.info("Brak zbuforowanych wynikow w %s - nic do wyslania", args.local_buffer_dir)
        return 0

    logger.info("Znaleziono %d zbuforowanych przebiegow w %s - probuje wyslac do Supabase", pending_before, args.local_buffer_dir)

    sent = local_buffer.flush_pending(repository)
    remaining = local_buffer.pending_count()

    if sent:
        logger.info("Wyslano %d/%d zbuforowanych przebiegow do Supabase", sent, pending_before)
    if remaining:
        logger.warning(
            "Nadal pozostaje %d zbuforowanych przebiegow (blad zapisu utrzymuje sie - sprawdz log powyzej "
            "z pelna trescia odpowiedzi Supabase). Napraw przyczyne i uruchom skrypt ponownie.",
            remaining,
        )
        return 1

    logger.info("Lokalny bufor pusty - wszystkie wyniki znajduja sie juz w Supabase")
    return 0


if __name__ == "__main__":
    sys.exit(main())
