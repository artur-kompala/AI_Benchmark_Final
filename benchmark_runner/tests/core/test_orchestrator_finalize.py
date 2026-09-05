"""Testy dla ExperimentRunner._finalize_and_persist - w szczegolnosci fallback
energy_joules_smart_plug (calkowanie probek) gdy watomierz nie zwrocil odczytu z
wlasnego licznika energii (np. urzadzenie bez odpowiedniego DPS, patrz
docs/setup_smart_plug.md)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.orchestrator import ExperimentRunner
from benchmark_runner.power.base_power_meter import PowerSample
from benchmark_runner.power.smart_plug.base import SmartPlugReading


class _FakeRepository:
    def __init__(self):
        self.devices = []
        self.runs = []
        self.power_samples = []
        self.summaries = []

    def get_or_create_device(self, device):
        self.devices.append(device)
        return "device-id"

    def insert_run(self, run):
        self.runs.append(run)
        return "run-id"

    def insert_power_samples_batch(self, samples):
        self.power_samples.extend(samples)

    def insert_run_summary(self, summary):
        self.summaries.append(summary)


class _FakeLocalBuffer:
    def enqueue_run_bundle(self, *args, **kwargs):
        raise AssertionError("nie powinno byc wywolane, gdy zapis do repo sie udaje")

    def flush_pending(self, repository):
        return 0


def _make_run_spec():
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[8],
        phases=["inference"],
    )
    return experiment.expand_matrix()[0]


def _make_device():
    return DeviceProfile(device_type="cpu", label="CPU", vendor=None, torch_device_str="cpu", backend="pytorch")


def test_finalize_falls_back_to_integration_when_no_smart_plug_reading():
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="smart_plug", watts=10.0),
        PowerSample(timestamp=t0 + timedelta(seconds=2), source="smart_plug", watts=20.0),
    ]
    # calkowanie trapezoidalne: (10 + 20) / 2 * 2s = 30 J

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=power_samples,
        smart_plug_reading=None,
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
    )

    assert len(repo.summaries) == 1
    assert repo.summaries[0].energy_joules_smart_plug == pytest.approx(30.0)


def test_finalize_prefers_device_reading_over_integration():
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="smart_plug", watts=10.0),
        PowerSample(timestamp=t0 + timedelta(seconds=2), source="smart_plug", watts=20.0),
    ]

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=power_samples,
        smart_plug_reading=SmartPlugReading(energy_wh=1.0),  # 1 Wh = 3600 J
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
    )

    assert repo.summaries[0].energy_joules_smart_plug == pytest.approx(3600.0)


def test_finalize_leaves_energy_none_when_no_reading_and_no_samples():
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=[],
        smart_plug_reading=None,
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
    )

    assert repo.summaries[0].energy_joules_smart_plug is None


def test_finalize_uses_codecarbon_when_rapl_has_no_samples():
    # Regresja Blad 2: rapl (pierwsze w cpu_sources) niedostepny na maszynie -> 0 probek,
    # codecarbon dziala poprawnie. avg_power_watts/energy_joules_software NIE moga byc
    # None, skoro codecarbon faktycznie dostarczyl dane - inaczej cala seria device=cpu
    # ma cicho bezuzyteczne dane mocy, mimo ze zrodlo dzialalo przez caly czas.
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="codecarbon", watts=45.0),
        PowerSample(timestamp=t0 + timedelta(seconds=2), source="codecarbon", watts=55.0),
    ]

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=power_samples,
        smart_plug_reading=None,
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
    )

    summary = repo.summaries[0]
    assert summary.avg_power_watts is not None
    assert summary.avg_power_watts == pytest.approx(50.0)
    assert summary.energy_joules_software is not None
    assert summary.power_samples_count == 2


def test_finalize_computes_flops_per_watt_from_flops_per_sample_and_avg_power():
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    # source="rapl" bo to preferred_source domyslny dla device_type="cpu"
    # (power_meter.cpu_sources=["rapl","codecarbon"]) - patrz _requested_sources().
    power_samples = [
        PowerSample(timestamp=t0, source="rapl", watts=100.0),
        PowerSample(timestamp=t0 + timedelta(seconds=2), source="rapl", watts=100.0),
    ]

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=power_samples,
        smart_plug_reading=None,
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
        flops_per_sample=2_000_000_000.0,
    )

    summary = repo.summaries[0]
    assert summary.avg_power_watts == pytest.approx(100.0)
    assert summary.flops_per_sample == pytest.approx(2_000_000_000.0)
    assert summary.flops_per_watt == pytest.approx(2_000_000_000.0 / 100.0)


def test_finalize_leaves_flops_per_watt_none_when_flops_per_sample_missing():
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="rapl", watts=100.0),
        PowerSample(timestamp=t0 + timedelta(seconds=2), source="rapl", watts=100.0),
    ]

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=power_samples,
        smart_plug_reading=None,
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
    )

    summary = repo.summaries[0]
    assert summary.flops_per_sample is None
    assert summary.flops_per_watt is None


def test_finalize_persists_accuracy_and_quantization_status():
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)

    runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="completed",
        error_message=None,
        samples_processed=10,
        power_samples=[],
        smart_plug_reading=None,
        duration_s=2.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=2),
        quantization_status="partial",
        accuracy=0.87,
    )

    summary = repo.summaries[0]
    assert summary.accuracy == pytest.approx(0.87)
    assert summary.quantization_status == "partial"


def test_finalize_logs_min_median_max_power_distribution(caplog):
    # Rozklad z wyraznym pojedynczym odstajacym niskim odczytem (10W wsrod probek ~26-27W) -
    # min/median musza pozwolic odroznic "jedna probka-odstajaca zanizyla srednia" od
    # "caly przebieg mial nizszy pobor mocy" (patrz zgloszenie: throttling/DVFS podejrzenie).
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="rapl", watts=26.0),
        PowerSample(timestamp=t0 + timedelta(seconds=1), source="rapl", watts=27.0),
        PowerSample(timestamp=t0 + timedelta(seconds=2), source="rapl", watts=10.0),
        PowerSample(timestamp=t0 + timedelta(seconds=3), source="rapl", watts=26.5),
    ]

    with caplog.at_level(logging.INFO, logger="benchmark_runner.core.orchestrator"):
        runner._finalize_and_persist(
            run_spec=_make_run_spec(),
            device=_make_device(),
            status="completed",
            error_message=None,
            samples_processed=10,
            power_samples=power_samples,
            smart_plug_reading=None,
            duration_s=4.0,
            started_at=t0,
            finished_at=t0 + timedelta(seconds=4),
        )

    distribution_logs = [r.message for r in caplog.records if "rozklad mocy" in r.message]
    assert len(distribution_logs) == 1
    message = distribution_logs[0]
    assert "min=10.00" in message
    assert "max=27.00" in message
    # median z [10, 26, 26.5, 27] (posortowane) = (26 + 26.5) / 2 = 26.25
    assert "median=26.25" in message


def test_finalize_rejects_physically_implausible_cpu_power_and_marks_run_partial(caplog):
    # Blad zgloszony w sesji: przy zablokowanym uprawnieniami RAPL, codecarbon w trybie
    # fallback potrafil zwrocic fizycznie niemozliwa moc CPU (300-1050W dla laptopa).
    # Nawet gdyby taka probka jakims cudem ominela pierwsza linie obrony w samych
    # samplerach (patrz tests/power/test_rapl_meter.py, test_codecarbon_meter.py),
    # _finalize_and_persist NIE MOZE cicho zapisac jej do Supabase.
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="rapl", watts=850.3),
        PowerSample(timestamp=t0 + timedelta(seconds=1), source="rapl", watts=920.1),
    ]

    with caplog.at_level(logging.WARNING, logger="benchmark_runner.core.orchestrator"):
        outcome = runner._finalize_and_persist(
            run_spec=_make_run_spec(),
            device=_make_device(),
            status="completed",
            error_message=None,
            samples_processed=10,
            power_samples=power_samples,
            smart_plug_reading=None,
            duration_s=1.0,
            started_at=t0,
            finished_at=t0 + timedelta(seconds=1),
        )

    summary = repo.summaries[0]
    assert summary.avg_power_watts is None
    assert summary.peak_power_watts is None
    assert summary.energy_joules_software is None
    assert summary.energy_per_sample_joules is None
    assert summary.energy_per_epoch_joules is None

    assert outcome.status == "partial"
    assert repo.runs[0].status == "partial"
    assert repo.runs[0].error_message is not None
    assert "150" in repo.runs[0].error_message

    warning_logs = [r.message for r in caplog.records if "zerowanie pol mocy" in r.message]
    assert len(warning_logs) == 1


def test_finalize_does_not_downgrade_already_failed_status():
    # Jesli przebieg juz zakonczyl sie status='failed' z innego powodu (np. wyjatek w
    # trakcie wykonania), implauzybilna moc NIE powinna "poprawiac" tego na 'partial' -
    # bardziej powazny status ma pierwszenstwo.
    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    t0 = datetime.now(timezone.utc)
    power_samples = [
        PowerSample(timestamp=t0, source="rapl", watts=850.3),
        PowerSample(timestamp=t0 + timedelta(seconds=1), source="rapl", watts=920.1),
    ]

    outcome = runner._finalize_and_persist(
        run_spec=_make_run_spec(),
        device=_make_device(),
        status="failed",
        error_message="oryginalny blad wykonania",
        samples_processed=10,
        power_samples=power_samples,
        smart_plug_reading=None,
        duration_s=1.0,
        started_at=t0,
        finished_at=t0 + timedelta(seconds=1),
    )

    assert outcome.status == "failed"
    assert repo.runs[0].error_message == "oryginalny blad wykonania"
    assert repo.summaries[0].avg_power_watts is None  # nadal odrzucone, tylko status sie nie zmienil
