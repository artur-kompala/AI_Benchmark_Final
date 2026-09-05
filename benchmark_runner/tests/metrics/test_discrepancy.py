from __future__ import annotations

import pytest

from benchmark_runner.metrics.discrepancy import compute_discrepancy, wh_to_joules


def test_compute_discrepancy_basic():
    # software=110J, smart_plug=100J -> 10% rozbieznosci
    assert compute_discrepancy(110.0, 100.0) == pytest.approx(10.0)


def test_compute_discrepancy_symmetric_absolute_value():
    assert compute_discrepancy(90.0, 100.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("software", "smart_plug"),
    [(None, 100.0), (100.0, None), (None, None)],
)
def test_compute_discrepancy_none_when_missing_values(software, smart_plug):
    assert compute_discrepancy(software, smart_plug) is None


def test_compute_discrepancy_none_on_zero_division():
    assert compute_discrepancy(100.0, 0.0) is None


def test_wh_to_joules():
    assert wh_to_joules(1.0) == pytest.approx(3600.0)
    assert wh_to_joules(0.5) == pytest.approx(1800.0)
