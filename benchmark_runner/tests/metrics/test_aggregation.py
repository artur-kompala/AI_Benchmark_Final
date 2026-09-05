from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from benchmark_runner.metrics.aggregation import aggregate_run, compute_throughput, integrate_energy_joules
from benchmark_runner.power.base_power_meter import PowerSample


def _make_samples(watts_sequence, interval_s: float = 1.0, source: str = "nvml") -> list[PowerSample]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        PowerSample(timestamp=start + timedelta(seconds=i * interval_s), source=source, watts=w)
        for i, w in enumerate(watts_sequence)
    ]


def test_integrate_energy_joules_constant_power():
    # stala moc 100W przez 10s (probki co 1s) -> energia = 100 * 10 = 1000 J
    samples = _make_samples([100.0] * 11, interval_s=1.0)
    assert integrate_energy_joules(samples) == pytest.approx(1000.0)


def test_integrate_energy_joules_ignores_none_watts():
    samples = _make_samples([100.0, None, 100.0], interval_s=1.0)
    # None jest pomijany - zostaja dwie probki 100W w odstepie 2s -> 200 J
    assert integrate_energy_joules(samples) == pytest.approx(200.0)


def test_integrate_energy_joules_insufficient_samples_returns_none():
    assert integrate_energy_joules([]) is None
    assert integrate_energy_joules(_make_samples([100.0])) is None


def test_aggregate_run_computes_per_sample_and_per_epoch_energy():
    samples = _make_samples([100.0] * 11, interval_s=1.0, source="nvml")
    result = aggregate_run(samples, samples_processed=100, num_epochs=2, preferred_source="nvml")

    assert result.energy_joules_software == pytest.approx(1000.0)
    assert result.energy_per_sample_joules == pytest.approx(10.0)
    assert result.energy_per_epoch_joules == pytest.approx(500.0)
    assert result.avg_power_watts == pytest.approx(100.0)
    assert result.peak_power_watts == pytest.approx(100.0)
    assert result.samples_processed == 100
    assert result.power_samples_count == 11


def test_aggregate_run_filters_by_preferred_source():
    nvml_samples = _make_samples([100.0] * 3, source="nvml")
    rapl_samples = _make_samples([50.0] * 3, source="rapl")
    result = aggregate_run(nvml_samples + rapl_samples, samples_processed=10, preferred_source="nvml")
    assert result.avg_power_watts == pytest.approx(100.0)
    # power_samples_count musi liczyc TYLKO probki z preferred_source (te ktore realnie
    # weszly do avg_power_watts), nie wszystkie zebrane probki ze wszystkich zrodel.
    assert result.power_samples_count == 3


def test_aggregate_run_without_preferred_source_uses_all_samples():
    samples = _make_samples([100.0] * 3, source="nvml")
    result = aggregate_run(samples, samples_processed=10, preferred_source=None)
    assert result.avg_power_watts == pytest.approx(100.0)
    assert result.power_samples_count == 3


def test_aggregate_run_power_samples_count_zero_when_no_samples():
    result = aggregate_run([], samples_processed=10, preferred_source="nvml")
    assert result.power_samples_count == 0
    assert result.avg_power_watts is None


def test_compute_throughput():
    assert compute_throughput(100, 10.0) == pytest.approx(10.0)
    assert compute_throughput(100, 0.0) is None
