from __future__ import annotations

import pytest

from benchmark_runner.power.smart_plug.manual_reader import ManualSmartPlugReader


def _make_input(responses: list[str]):
    it = iter(responses)

    def input_fn(prompt: str) -> str:
        return next(it)

    return input_fn


def test_manual_reader_computes_energy_delta():
    reader = ManualSmartPlugReader(
        enabled=True, input_fn=_make_input(["100.0", "150.5"]), output_fn=lambda msg: None
    )
    reader.start_run("run-1")
    reading = reader.end_run("run-1")
    assert reading is not None
    assert reading.energy_wh == pytest.approx(50.5)


def test_manual_reader_disabled_returns_none():
    reader = ManualSmartPlugReader(enabled=False, input_fn=_make_input([]), output_fn=lambda msg: None)
    reader.start_run("run-1")
    assert reader.end_run("run-1") is None


def test_manual_reader_skipped_start_returns_none_end():
    reader = ManualSmartPlugReader(enabled=True, input_fn=_make_input([""]), output_fn=lambda msg: None)
    reader.start_run("run-1")  # pusty input -> pomija odczyt startowy
    assert reader.end_run("run-1") is None


def test_manual_reader_accepts_comma_decimal_separator():
    reader = ManualSmartPlugReader(
        enabled=True, input_fn=_make_input(["100,0", "120,5"]), output_fn=lambda msg: None
    )
    reader.start_run("run-1")
    reading = reader.end_run("run-1")
    assert reading.energy_wh == pytest.approx(20.5)


def test_manual_reader_rejects_negative_delta():
    reader = ManualSmartPlugReader(
        enabled=True, input_fn=_make_input(["100.0", "50.0"]), output_fn=lambda msg: None
    )
    reader.start_run("run-1")
    assert reader.end_run("run-1") is None


def test_manual_reader_reprompts_on_invalid_input():
    reader = ManualSmartPlugReader(
        enabled=True,
        input_fn=_make_input(["not-a-number", "100.0", "150.0"]),
        output_fn=lambda msg: None,
    )
    reader.start_run("run-1")
    reading = reader.end_run("run-1")
    assert reading.energy_wh == pytest.approx(50.0)
