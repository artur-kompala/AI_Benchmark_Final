"""Testy LocalBuffer: symulacja braku sieci -> zapis do bufora -> flush po 'odzyskaniu' polaczenia."""

from __future__ import annotations

from datetime import datetime, timezone

from benchmark_runner.storage.local_buffer import LocalBuffer
from benchmark_runner.storage.models import DeviceRecord, PowerSampleRecord, RunRecord, RunSummaryRecord
from benchmark_runner.storage.repository import RepositoryError


class _FailingRepository:
    def get_or_create_device(self, device):
        raise RepositoryError("brak polaczenia (test)")


class _SucceedingRepository:
    def __init__(self) -> None:
        self.saved_runs: list[RunRecord] = []

    def get_or_create_device(self, device):
        return "device-xyz"

    def insert_run(self, run):
        self.saved_runs.append(run)
        return "run-xyz"

    def insert_power_samples_batch(self, samples):
        for s in samples:
            assert s.run_id == "run-xyz"

    def insert_run_summary(self, summary):
        assert summary.run_id == "run-xyz"


def _sample_bundle_args():
    now = datetime.now(timezone.utc)
    device = DeviceRecord(machine_name="PC1", device_type="cpu", device_label="CPU")
    run = RunRecord(
        device_id="",
        task="image_classification",
        model_name="mobilenet_v3",
        phase="train",
        precision="fp32",
        config_hash="hash1",
        config_snapshot={"a": 1},
        started_at=now,
        finished_at=now,
        duration_s=1.0,
    )
    power_samples = [PowerSampleRecord(run_id="", timestamp=now, source="nvml", watts=100.0)]
    summary = RunSummaryRecord(run_id="")
    return device, run, power_samples, summary


def test_enqueue_writes_to_buffer_file(tmp_path):
    buffer = LocalBuffer(buffer_dir=tmp_path)
    bundle_id = buffer.enqueue_run_bundle(*_sample_bundle_args())

    assert bundle_id
    assert buffer.pending_count() == 1


def test_flush_pending_with_failing_repository_keeps_bundle(tmp_path):
    buffer = LocalBuffer(buffer_dir=tmp_path)
    buffer.enqueue_run_bundle(*_sample_bundle_args())

    sent = buffer.flush_pending(_FailingRepository())

    assert sent == 0
    assert buffer.pending_count() == 1


def test_flush_pending_with_succeeding_repository_empties_buffer(tmp_path):
    buffer = LocalBuffer(buffer_dir=tmp_path)
    buffer.enqueue_run_bundle(*_sample_bundle_args())

    repo = _SucceedingRepository()
    sent = buffer.flush_pending(repo)

    assert sent == 1
    assert buffer.pending_count() == 0
    assert len(repo.saved_runs) == 1


def test_flush_pending_empty_buffer_returns_zero(tmp_path):
    buffer = LocalBuffer(buffer_dir=tmp_path)
    assert buffer.flush_pending(_SucceedingRepository()) == 0
