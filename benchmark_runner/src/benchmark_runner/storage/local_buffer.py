"""Lokalny bufor/kolejka na wypadek braku internetu podczas serii pomiarowej.

Gdy zapis do Supabase sie nie powiedzie (RepositoryError), orchestrator buforuje caly
"pakiet" wynikow jednego przebiegu (device + run + power_samples + summary) do pliku
JSONL na dysku i probuje ponownie wyslac wszystkie zalegle pakiety przy nastepnej okazji
(np. na koniec sesji) - utrata sieci w trakcie serii nie powoduje utraty wynikow.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_runner.storage.models import (
    DeviceRecord,
    PowerSampleRecord,
    RunRecord,
    RunSummaryRecord,
)
from benchmark_runner.storage.repository import RepositoryError, RunRepository

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_FILENAME = "pending_runs.jsonl"


@dataclass
class RunBundle:
    """Kompletny wynik jednego przebiegu, jeszcze niezapisany w Supabase."""

    bundle_id: str
    device: DeviceRecord
    run: RunRecord
    power_samples: list[PowerSampleRecord]
    summary: RunSummaryRecord | None


def _serialize_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, dict):
        return {k: _serialize_datetime(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_datetime(v) for v in value]
    return value


def _deserialize_datetime(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"__datetime__"}:
            return datetime.fromisoformat(value["__datetime__"])
        return {k: _deserialize_datetime(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_datetime(v) for v in value]
    return value


class LocalBuffer:
    def __init__(self, buffer_dir: str | Path) -> None:
        self._buffer_dir = Path(buffer_dir)
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_path = self._buffer_dir / DEFAULT_BUFFER_FILENAME

    def enqueue_run_bundle(
        self,
        device: DeviceRecord,
        run: RunRecord,
        power_samples: list[PowerSampleRecord],
        summary: RunSummaryRecord | None,
    ) -> str:
        bundle = RunBundle(
            bundle_id=str(uuid.uuid4()),
            device=device,
            run=run,
            power_samples=power_samples,
            summary=summary,
        )
        line = json.dumps(_serialize_datetime(asdict(bundle)), ensure_ascii=False)
        with self._buffer_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.warning(
            "Zapis do Supabase niedostepny - wynik przebiegu zbuforowany lokalnie (%s), bundle_id=%s",
            self._buffer_path,
            bundle.bundle_id,
        )
        return bundle.bundle_id

    def _read_pending(self) -> list[dict[str, Any]]:
        if not self._buffer_path.exists():
            return []
        lines = self._buffer_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def pending_count(self) -> int:
        return len(self._read_pending())

    def flush_pending(self, repository: RunRepository) -> int:
        """Probuje wyslac wszystkie zbuforowane pakiety w kolejnosci FIFO. Zwraca liczbe
        pomyslnie wyslanych. Zatrzymuje sie przy pierwszym bledzie (siec dalej niedostepna),
        pozostawiajac reszte w buforze do kolejnej proby."""
        pending = self._read_pending()
        if not pending:
            return 0

        sent = 0
        remaining = list(pending)
        for raw_bundle in pending:
            try:
                self._replay_bundle(raw_bundle, repository)
            except RepositoryError as exc:
                logger.warning(
                    "Nadal brak polaczenia z Supabase - przerywam flush bufora lokalnego (%d/%d wyslanych). Przyczyna: %s",
                    sent,
                    len(pending),
                    exc,
                )
                break
            remaining.pop(0)
            sent += 1

        self._rewrite_buffer(remaining)
        if sent:
            logger.info("Wyslano %d zbuforowanych przebiegow z lokalnego bufora do Supabase", sent)
        return sent

    def _replay_bundle(self, raw_bundle: dict[str, Any], repository: RunRepository) -> None:
        data = _deserialize_datetime(raw_bundle)

        device = DeviceRecord(**data["device"])
        run_data = data["run"]
        power_samples_data = data["power_samples"]
        summary_data = data["summary"]

        device_id = repository.get_or_create_device(device)
        run_data["device_id"] = device_id
        run_id = repository.insert_run(RunRecord(**run_data))

        power_samples = []
        for ps in power_samples_data:
            ps["run_id"] = run_id
            power_samples.append(PowerSampleRecord(**ps))
        repository.insert_power_samples_batch(power_samples)

        if summary_data is not None:
            summary_data["run_id"] = run_id
            repository.insert_run_summary(RunSummaryRecord(**summary_data))

    def _rewrite_buffer(self, remaining: list[dict[str, Any]]) -> None:
        if not remaining:
            if self._buffer_path.exists():
                self._buffer_path.unlink()
            return
        with self._buffer_path.open("w", encoding="utf-8") as f:
            for bundle in remaining:
                f.write(json.dumps(bundle, ensure_ascii=False) + "\n")
