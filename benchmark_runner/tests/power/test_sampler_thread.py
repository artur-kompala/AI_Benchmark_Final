from __future__ import annotations

import time
from datetime import datetime, timezone

from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample
from benchmark_runner.power.sampler_thread import PowerSamplingThread


class _FakeSampler(BasePowerSampler):
    source_name = "fake"

    def __init__(self) -> None:
        self.closed = False

    def is_available(self) -> bool:
        return True

    def sample(self) -> PowerSample:
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=42.0)

    def close(self) -> None:
        self.closed = True


class _UnavailableSampler(BasePowerSampler):
    source_name = "unavailable"

    def is_available(self) -> bool:
        return False

    def sample(self) -> PowerSample:
        raise AssertionError("sample() nie powinno byc wywolane dla niedostepnego samplera")


class _FlakySampler(BasePowerSampler):
    source_name = "flaky"

    def is_available(self) -> bool:
        return True

    def sample(self) -> PowerSample:
        raise RuntimeError("blad odczytu (test)")


def test_sampler_thread_collects_samples_from_available_sampler():
    sampler = _FakeSampler()
    thread = PowerSamplingThread([sampler], interval_ms=10).start()
    time.sleep(0.1)
    samples = thread.stop()

    assert len(samples) > 0
    assert all(s.source == "fake" for s in samples)
    assert sampler.closed is True


def test_sampler_thread_skips_unavailable_sampler():
    thread = PowerSamplingThread([_UnavailableSampler()], interval_ms=10).start()
    time.sleep(0.05)
    samples = thread.stop()
    assert samples == []


def test_sampler_thread_continues_after_sample_error():
    thread = PowerSamplingThread([_FlakySampler()], interval_ms=10).start()
    time.sleep(0.05)
    samples = thread.stop()  # nie powinno rzucic wyjatku mimo bledow w sample()
    assert samples == []
