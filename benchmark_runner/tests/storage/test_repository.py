"""Testy RunRepository z mockiem klienta Supabase (bez prawdziwego polaczenia sieciowego)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from benchmark_runner.storage.models import DeviceRecord, RunRecord, RunSummaryRecord
from benchmark_runner.storage.repository import RepositoryError, RunRepository


class _FakeResult:
    def __init__(self, data) -> None:
        self.data = data


class _FakeQuery:
    """Emuluje lancuch wywolan supabase-py: table(...).select(...).eq(...).limit(...).execute()."""

    def __init__(self, execute_result, raise_on_execute: Exception | None = None) -> None:
        self._execute_result = execute_result
        self._raise_on_execute = raise_on_execute

    def select(self, *args, **kwargs):
        return self

    def insert(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        if self._raise_on_execute:
            raise self._raise_on_execute
        return _FakeResult(self._execute_result)


class _FakeClient:
    """Klient, ktory zawsze zwraca ten sam zaprogramowany wynik dla danej tabeli."""

    def __init__(self, table_results: dict[str, list], raise_on_table: str | None = None) -> None:
        self._table_results = table_results
        self._raise_on_table = raise_on_table

    def table(self, name: str) -> _FakeQuery:
        if name == self._raise_on_table:
            return _FakeQuery(None, raise_on_execute=ConnectionError("brak polaczenia (test)"))
        return _FakeQuery(self._table_results.get(name, []))


class _SequencedClient:
    """Zwraca kolejne zaprogramowane wyniki execute() przy kolejnych wywolaniach .table(...) -
    przydatne dla get_or_create_device, ktore woloa .table('devices') dwukrotnie (select, insert)."""

    def __init__(self, results_sequence: list[list]) -> None:
        self._results = iter(results_sequence)

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(next(self._results))


def _run_record() -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        device_id="device-1",
        task="image_classification",
        model_name="mobilenet_v3",
        phase="train",
        precision="fp32",
        config_hash="abc123",
        config_snapshot={"foo": "bar"},
        started_at=now,
        finished_at=now,
        duration_s=1.23,
    )


def test_get_or_create_device_returns_existing_id():
    client = _FakeClient(table_results={"devices": [{"id": "existing-device-id"}]})
    repo = RunRepository(client)
    device = DeviceRecord(machine_name="PC1", device_type="cpu", device_label="CPU")
    assert repo.get_or_create_device(device) == "existing-device-id"


def test_get_or_create_device_creates_when_missing():
    # 1. select -> pusta lista (brak istniejacego urzadzenia)
    # 2. insert -> nowe urzadzenie z id
    client = _SequencedClient([[], [{"id": "new-device-id"}]])
    repo = RunRepository(client)
    device = DeviceRecord(machine_name="PC1", device_type="cpu", device_label="CPU")
    assert repo.get_or_create_device(device) == "new-device-id"


def test_insert_run_returns_id():
    client = _FakeClient(table_results={"runs": [{"id": "run-1"}]})
    repo = RunRepository(client)
    assert repo.insert_run(_run_record()) == "run-1"


def test_insert_power_samples_batch_empty_list_is_noop():
    client = _FakeClient(table_results={})
    repo = RunRepository(client)
    repo.insert_power_samples_batch([])  # nie powinno rzucic wyjatku ani nic wysylac


def test_insert_run_summary_success():
    client = _FakeClient(table_results={"run_summary": [{"id": "summary-1"}]})
    repo = RunRepository(client)
    repo.insert_run_summary(RunSummaryRecord(run_id="run-1"))


def test_repository_raises_repository_error_on_connection_failure():
    client = _FakeClient(table_results={}, raise_on_table="runs")
    repo = RunRepository(client)
    with pytest.raises(RepositoryError):
        repo.insert_run(_run_record())
