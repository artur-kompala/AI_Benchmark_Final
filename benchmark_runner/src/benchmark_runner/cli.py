"""CLI runnera: python -m benchmark_runner run --config <path> [--device ...] [--dry-run]"""

from __future__ import annotations

import argparse
import logging
import sys

from benchmark_runner.core.orchestrator import ExperimentRunner
from benchmark_runner.storage.local_buffer import LocalBuffer
from benchmark_runner.storage.repository import RunRepository
from benchmark_runner.storage.supabase_client import SupabaseConfigError, get_supabase_client
from benchmark_runner.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_runner", description="Headless CLI runner benchmarkow CPU/GPU/NPU"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Uruchom eksperyment z pliku konfiguracji YAML")
    run_parser.add_argument("--config", required=True, help="Sciezka do pliku konfiguracji eksperymentu (YAML)")
    run_parser.add_argument(
        "--device",
        default=None,
        help="Nadpisuje urzadzenie z configu dla wszystkich przebiegow (np. cpu, cuda, rocm)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Smoke test: mala liczba epok/iteracji (1 epoka, 3 iteracje inferencji), "
            "zeby szybko zweryfikowac caly przeplyw - wynik nadal jest zapisywany do Supabase"
        ),
    )
    run_parser.add_argument(
        "--local-buffer-dir",
        default="./.local_buffer/",
        help="Katalog na lokalny bufor wynikow przy braku internetu (domyslnie ./.local_buffer/)",
    )
    run_parser.add_argument(
        "--no-smart-plug",
        action="store_true",
        help=(
            "Wylacza watomierz fizyczny dla calej serii, NIEZALEZNIE od configu YAML. "
            "Przydatne na maszynach bez wlasnego watomierza Tuya - bez tej flagi tryb "
            "'manual' (fallback gdy backend='tuya' ale brak danych logowania w .env) "
            "zatrzymuje sie przed KAZDYM przebiegiem i czeka na reczny odczyt Wh z "
            "klawiatury, co przy dlugiej serii bez nadzoru zawiesi cala sesje."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)

    parser.print_help()
    return 1


def _run_command(args: argparse.Namespace) -> int:
    try:
        client = get_supabase_client()
    except SupabaseConfigError as exc:
        logger.error(str(exc))
        return 1

    repository = RunRepository(client)
    local_buffer = LocalBuffer(buffer_dir=args.local_buffer_dir)
    runner = ExperimentRunner(repository=repository, local_buffer=local_buffer)

    try:
        runner.run_from_config(
            args.config,
            device_override=args.device,
            dry_run=args.dry_run,
            no_smart_plug=args.no_smart_plug,
        )
    except Exception:
        logger.error("Nieoczekiwany blad podczas uruchamiania serii eksperymentow", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
