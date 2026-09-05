"""Testy _maybe_scale_up_num_iterations (Problem 1 - okno pomiaru mocy krotsze niz
sampling_interval_ms). Przy bardzo krotkich przebiegach (np. batch_size=1) sampler_thread
zbiera zbyt malo probek mocy w calym oknie pomiaru, przez co avg_power_watts to
praktycznie pojedynczy losowy odczyt zamiast realnej sredniej (potwierdzone empirycznie:
3 powtorzenia tej samej konfiguracji dawaly 17.6W/7.3W/15.4W). Ta funkcja szacuje czas
trwania na podstawie zmierzonego warmup i proaktywnie podnosi num_iterations, zeby zebrac
wystarczajaco duzo probek."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pytest

import math

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.orchestrator import (
    _MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S,
    _MIN_POWER_SAMPLES_THRESHOLD,
    _maybe_scale_up_num_iterations,
)
from benchmark_runner.metrics.aggregation import aggregate_run
from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample
from benchmark_runner.power.sampler_thread import PowerSamplingThread


class _AlwaysAvailableFakeSampler(BasePowerSampler):
    """Zrodlo mocy zawsze dostepne, zwraca stala moc przy kazdym odczycie - uzywane do
    zweryfikowania, ze PowerSamplingThread faktycznie zbiera probki w oknie czasowym
    wyznaczonym przez PRZESKALOWANE (przez _maybe_scale_up_num_iterations) num_iterations,
    a nie w oknie sprzed skalowania (regresja: brak race condition miedzy przeliczeniem
    scalingu a startem watku probkujacego)."""

    source_name = "fake_always_available"

    def is_available(self) -> bool:
        return True

    def sample(self) -> PowerSample:
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=42.0)


def _make_run_spec(*, warmup_iterations=5, num_iterations=50, sampling_interval_ms=100):
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[1],
        phases=["inference"],
    )
    run_spec = experiment.expand_matrix()[0]
    run_spec.warmup.iterations = warmup_iterations
    run_spec.inference.num_iterations = num_iterations
    run_spec.power_meter.sampling_interval_ms = sampling_interval_ms
    return run_spec


def test_scales_up_when_expected_samples_below_threshold():
    # per_iter_s=0.05, sampling_interval_s=0.1 -> target=ceil(25*0.1/0.05)=50
    # configured=10 -> estimated_duration=0.5s -> expected_samples=5 < 20 -> musi podniesc
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=10, sampling_interval_ms=100)
    warmup_duration_s = 0.05 * 5  # 0.25s dla 5 iteracji warmup -> per_iter_s=0.05

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    assert run_spec.inference.num_iterations == 50


def test_does_not_scale_when_already_enough_samples():
    # per_iter_s=0.05, configured=50 -> estimated_duration=2.5s -> expected_samples=25 >= 20
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=50, sampling_interval_ms=100)
    warmup_duration_s = 0.05 * 5

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    assert run_spec.inference.num_iterations == 50


def test_boundary_exactly_at_threshold_does_not_scale():
    # per_iter_s=0.1, sampling_interval_s=0.1, configured=20 -> expected_samples dokladnie 20
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=20, sampling_interval_ms=100)
    warmup_duration_s = 0.1 * 5

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    assert run_spec.inference.num_iterations == 20


def test_never_decreases_num_iterations():
    # per_iter_s=0.02, configured=1000 -> estimated_duration=20s -> expected_samples=200 >= 20
    # -> juz wystarczajaco probek, funkcja nie powinna nic zmieniac (na pewno nie zmniejszyc).
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=1000, sampling_interval_ms=100)
    warmup_duration_s = 0.02 * 5

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    assert run_spec.inference.num_iterations == 1000


def test_scaling_is_capped_at_max_cumulative_duration():
    # sampling_interval_ms=2000 (2s) -> zebranie 25 probek (_MIN_POWER_SAMPLES_TARGET)
    # matematycznie wymagaloby 50s pomiaru, znacznie ponad limit czasowy (20s) - musi byc
    # przyciete do floor(_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S / per_iter_s), NIE do
    # jakiejs wielokrotnosci num_iterations z configu.
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=5, sampling_interval_ms=2000)
    per_iter_s = 0.001
    warmup_duration_s = per_iter_s * 5

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    expected = math.floor(_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S / per_iter_s)
    assert run_spec.inference.num_iterations == expected
    # Zdroworozsadkowa asercja niezalezna od dokladnej wartosci stalej: przy tak duzym
    # sampling_interval, przeskalowana liczba iteracji * per_iter_s NIE MOZE przekroczyc
    # limitu czasowego wiecej niz nieznacznie (blad zaokraglenia floor/ceil).
    assert run_spec.inference.num_iterations * per_iter_s <= _MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S


def test_noop_when_warmup_disabled_zero_iterations():
    run_spec = _make_run_spec(warmup_iterations=0, num_iterations=10, sampling_interval_ms=100)

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s=0.0)

    assert run_spec.inference.num_iterations == 10


def test_noop_when_warmup_duration_is_zero():
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=10, sampling_interval_ms=100)

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s=0.0)

    assert run_spec.inference.num_iterations == 10


def test_noop_when_sampling_interval_is_zero():
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=10, sampling_interval_ms=0)

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s=0.25)

    assert run_spec.inference.num_iterations == 10


def test_reported_empirical_scenario_batch_size_1():
    # Scenariusz ze zgloszenia: batch_size=1, duration_s~0.14s, num_iterations=50 (domyslne),
    # sampling_interval_ms=100 -> ~1.4 probki (ponizej progu) - musi podniesc num_iterations.
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=50, sampling_interval_ms=100)
    warmup_duration_s = 0.014  # 5 iteracji warmup, ten sam rzad wielkosci co ~0.14s/50 iter

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    assert run_spec.inference.num_iterations > 50
    per_iter_s = warmup_duration_s / 5
    new_estimated_duration_s = per_iter_s * run_spec.inference.num_iterations
    new_expected_samples = new_estimated_duration_s / 0.1
    assert new_expected_samples >= _MIN_POWER_SAMPLES_THRESHOLD


# ============================================================
# Regresja: po zadzialaniu auto-scalingu, PowerSamplingThread uruchomiony DOPIERO PO
# przeliczeniu scalingu (jak w core/orchestrator.py::_run_single) musi faktycznie zebrac
# probki mocy, jesli jakiekolwiek zrodlo jest dostepne - zero probek mimo dostepnego
# zrodla jest bledem, ktory ma wykrywac CI, nie tylko reczna inspekcja danych w Supabase.
# Test uruchamia PRAWDZIWY watek (realny czas, nie mock time.sleep) - zamierzone: to jest
# dokladnie ten mechanizm (wielowatkowosc + timing), ktory mial powodowac podejrzewana
# race condition.
# ============================================================


def test_power_sampling_thread_collects_samples_over_scaled_duration():
    run_spec = _make_run_spec(warmup_iterations=1, num_iterations=3, sampling_interval_ms=20)
    per_iter_s = 0.02
    warmup_duration_s = per_iter_s * run_spec.warmup.iterations

    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)
    assert run_spec.inference.num_iterations > 3, "scaling musi zadzialac, inaczej test niczego nie sprawdza"

    sampler = _AlwaysAvailableFakeSampler()
    thread = PowerSamplingThread([sampler], interval_ms=run_spec.power_meter.sampling_interval_ms).start()
    # Symuluje faktyczny czas trwania POMIARU PO scalingu (dokladnie ta wartosc, ktora w
    # _run_single trafia do task.run_inference_pass() i realnie okresla, jak dlugo dziala
    # watek probkujacy przed sampling_thread.stop()).
    time.sleep(per_iter_s * run_spec.inference.num_iterations)
    samples = thread.stop()

    assert len(samples) > 0, (
        "PowerSamplingThread nie zebral ani jednej probki mimo dostepnego zrodla i "
        "przeskalowanego czasu trwania - to regresja (np. race condition miedzy "
        "scalingiem a startem watku), nie brak dostepnego zrodla mocy"
    )

    metrics = aggregate_run(
        samples,
        samples_processed=run_spec.inference.num_iterations,
        preferred_source=sampler.source_name,
    )
    assert metrics.power_samples_count > 0
    assert metrics.avg_power_watts is not None
    assert metrics.avg_power_watts == pytest.approx(42.0)


def test_scaling_log_message_formats_without_error(caplog):
    # Regresja: logger.info() z formatem %-placeholder musi dostac dokladnie tyle
    # argumentow, ile ma placeholderow w stringu - brakujacy argument nie rzuca wyjatku
    # w miejscu wywolania (logging polyka to wewnetrznie i drukuje "--- Logging error ---"
    # na stderr), wiec latwo to przeoczyc bez jawnego testu wymuszajacego getMessage().
    run_spec = _make_run_spec(warmup_iterations=5, num_iterations=50, sampling_interval_ms=100)
    warmup_duration_s = 0.07332075800013627  # odtwarza dokladnie zgloszony scenariusz (~7.3 probki)

    with caplog.at_level(logging.INFO, logger="benchmark_runner.core.orchestrator"):
        _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

    assert run_spec.inference.num_iterations == 171
    scaling_records = [r for r in caplog.records if "zbyt krotki" in r.message]
    assert len(scaling_records) == 1
    # getMessage() wykonuje faktyczne formatowanie (msg %% args) - rzuci TypeError przy
    # niezgodnej liczbie argumentow, dokladnie tak jak zrobil to prawdziwy handler logowania.
    formatted = scaling_records[0].getMessage()
    assert ">=25 probek mocy" in formatted
