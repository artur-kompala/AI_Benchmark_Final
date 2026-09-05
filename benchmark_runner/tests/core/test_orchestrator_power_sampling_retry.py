"""Test regresyjny end-to-end dla retry loop w ExperimentRunner._run_single (Problem 1 z
kolejnego zgloszenia): po przeskalowaniu num_iterations na podstawie warmup (~171 z ~50),
faktyczna liczba probek mocy nadal wynosila 4-14 zamiast ~25 - warmup systematycznie
zawyza per-iteration time (pierwsze wywolanie po kompilacji OpenVINO bywa nieproporcjonalnie
wolne), wiec _maybe_scale_up_num_iterations() niedoszacowuje. Retry loop w _run_single musi
to skorygowac na podstawie NAPRAWDE zmierzonego czasu proby, nie oszacowania z warmup.

Symulujemy dokladnie ten efekt: warmup() jest SZTUCZNIE 10x wolniejszy per-iteracja niz
run_inference_pass() - _maybe_scale_up_num_iterations() policzy wiec za malo iteracji na
pierwsza probe, a test sprawdza, ze mimo to finalny power_samples_count osiaga prog."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core import orchestrator as orchestrator_module
from benchmark_runner.core.base_task import BaseTask, TaskStepResult
from benchmark_runner.core.orchestrator import (
    _MIN_POWER_SAMPLES_THRESHOLD,
    ExperimentRunner,
)
from benchmark_runner.core.registry import TASK_REGISTRY, register_task
from benchmark_runner.power.base_power_meter import BasePowerSampler, PowerSample

_TASK_NAME = "test_retry_loop_task"


class _MisleadingWarmupTask(BaseTask):
    _TRUE_PER_ITER_S = 0.002  # prawdziwy (szybszy) koszt per iteracje w run_inference_pass()
    _WARMUP_PER_ITER_S = 0.02  # 10x wolniej - symuluje myslacy warmup po kompilacji OpenVINO

    def prepare(self) -> None:
        pass

    def warmup(self, n_iters: int) -> None:
        time.sleep(self._WARMUP_PER_ITER_S * n_iters)

    def run_inference_pass(self) -> TaskStepResult:
        num_iterations = self.run_context.run_spec.inference.num_iterations
        time.sleep(self._TRUE_PER_ITER_S * num_iterations)
        return TaskStepResult(samples_processed=num_iterations, accuracy=None)


class _FakeAlwaysAvailableSampler(BasePowerSampler):
    source_name = "fake_test_source"

    def is_available(self) -> bool:
        return True

    def sample(self) -> PowerSample:
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=10.0)


class _FakeRepository:
    def __init__(self) -> None:
        self.summaries = []

    def get_or_create_device(self, device):
        return "device-id"

    def insert_run(self, run):
        return "run-id"

    def insert_power_samples_batch(self, samples):
        pass

    def insert_run_summary(self, summary):
        self.summaries.append(summary)


class _FakeLocalBuffer:
    def enqueue_run_bundle(self, *args, **kwargs):
        raise AssertionError("zapis do fake repo zawsze sie udaje - buforowanie nie powinno byc wywolane")

    def flush_pending(self, repository):
        return 0


def _make_run_spec(*, num_iterations=50, warmup_iterations=5, sampling_interval_ms=20):
    experiment = ExperimentConfig(
        experiment_name="test",
        task=_TASK_NAME,
        dataset=None,
        models=["dummy"],
        devices=["cpu"],
        precisions=["fp32"],
        batch_sizes=[1],
        phases=["inference"],
    )
    run_spec = experiment.expand_matrix()[0]
    run_spec.warmup.iterations = warmup_iterations
    run_spec.inference.num_iterations = num_iterations
    run_spec.power_meter.sampling_interval_ms = sampling_interval_ms
    # Kieruje preferred_source (patrz _requested_sources w orchestrator.py) na nasz fejkowy
    # sampler, zamiast domyslnego rapl/codecarbon (ktorych i tak nie uzywamy - build_power_
    # samplers jest ponizej podmienione na fejkowe zrodlo).
    run_spec.power_meter.cpu_sources = ["fake_test_source"]
    # Wylacza watomierz fizyczny (domyslnie enabled=True, backend='manual') - inaczej test
    # zawiesilby sie na input() (patrz ManualSmartPlugReader), tak jak --no-smart-plug w cli.py.
    run_spec.power_meter.smart_plug.enabled = False
    return run_spec


@pytest.fixture(autouse=True)
def _register_fake_task():
    if _TASK_NAME not in TASK_REGISTRY:
        register_task(_TASK_NAME)(_MisleadingWarmupTask)
    yield


def test_retry_loop_corrects_misleading_warmup_estimate(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module, "build_power_samplers", lambda device, power_meter: [_FakeAlwaysAvailableSampler()]
    )

    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())
    run_spec = _make_run_spec(num_iterations=50, warmup_iterations=5, sampling_interval_ms=20)

    outcome = runner._run_single(run_spec, device_override="cpu")

    assert outcome.status == "completed"
    assert len(repo.summaries) == 1
    summary = repo.summaries[0]

    assert summary.power_samples_count is not None
    assert summary.power_samples_count >= _MIN_POWER_SAMPLES_THRESHOLD, (
        f"Retry loop nie osiagnal progu {_MIN_POWER_SAMPLES_THRESHOLD} probek mocy mimo "
        f"dostepnego zrodla i wielu prob (zebrano {summary.power_samples_count})"
    )
    assert summary.avg_power_watts == pytest.approx(10.0)
    # num_iterations faktycznie uzyte w ostatniej probie musialo wzrosnac ponad
    # oszacowanie z (mylacego) warmup, zeby osiagnac prog - potwierdza ze korekta na
    # podstawie realnego czasu zadzialala, nie tylko jednorazowe oszacowanie z warmup.
    assert run_spec.inference.num_iterations > 50


# ============================================================
# Problem 3: zrodlo o duzym, JEDNORAZOWYM narzucie startowym (np. codecarbon -
# EmissionsTracker.is_available()=~1.4s + PIERWSZY flush()=~3.1s, zmierzone empirycznie
# na maszynie z tego projektu) w polaczeniu z zadaniem o bardzo wysokiej przepustowosci
# (krotki czas pojedynczej iteracji). Poprzednia wersja formuly retry liczyla kazda probe
# tak, jakby probkowanie zaczynalo sie od zera - przy stalym per-iteration time dawalo to
# ZAWSZE ten sam "docelowy" num_iterations, wiec po jednym kroku retry loop uznawal to za
# brak postepu i przerywal, MIMO ZE realnie zebrana liczba probek nadal wynosila 0 (caly
# dotychczasowy czas zjadl jednorazowy narzut startowy zrodla, ktore - co wazne - NIE jest
# restartowane miedzy probami, tylko jego PIERWSZE wywolanie sample() jest po prostu
# bardzo wolne). Naprawiona wersja liczy DODATKOWE iteracje na podstawie SKUMULOWANEGO,
# nieprzerwanego czasu pomiaru ze wszystkich dotychczasowych prob.
# ============================================================


class _ConstantRateTask(BaseTask):
    """Zadanie o stalej, bardzo szybkiej przepustowosci per iteracje (ten sam czas dla
    warmup i inferencji - w odroznieniu od _MisleadingWarmupTask wyzej, ktora celowo
    testuje INNY blad/mechanizm) - reprezentuje "bardzo wysoka przepustowosc, krotki czas
    pojedynczej iteracji" ze zgloszenia."""

    PER_ITER_S = 0.0002

    def prepare(self) -> None:
        pass

    def warmup(self, n_iters: int) -> None:
        time.sleep(self.PER_ITER_S * n_iters)

    def run_inference_pass(self) -> TaskStepResult:
        num_iterations = self.run_context.run_spec.inference.num_iterations
        time.sleep(self.PER_ITER_S * num_iterations)
        return TaskStepResult(samples_processed=num_iterations, accuracy=None)


class _SlowStartFakeSampler(BasePowerSampler):
    """Symuluje zrodlo o duzym, jednorazowym narzucie startowym PIERWSZEGO wywolania
    sample() (jak codecarbon empirycznie na tej maszynie), po ktorym kolejne wywolania sa
    juz szybkie i regularne. Sampler NIE jest restartowany miedzy probami retry loop (to
    ta sama instancja przez caly _run_single) - narzut placi sie wiec dokladnie raz."""

    source_name = "fake_slow_start_source"

    def __init__(self, startup_delay_s: float) -> None:
        self._startup_delay_s = startup_delay_s
        self._first_call_done = False

    def is_available(self) -> bool:
        return True

    def sample(self) -> PowerSample:
        if not self._first_call_done:
            time.sleep(self._startup_delay_s)
            self._first_call_done = True
        return PowerSample(timestamp=datetime.now(timezone.utc), source=self.source_name, watts=10.0)


_TASK_NAME_CONSTANT_RATE = "test_retry_loop_task_constant_rate"


@pytest.fixture(autouse=True)
def _register_constant_rate_task():
    if _TASK_NAME_CONSTANT_RATE not in TASK_REGISTRY:
        register_task(_TASK_NAME_CONSTANT_RATE)(_ConstantRateTask)
    yield


def _make_high_throughput_run_spec(*, num_iterations=50, warmup_iterations=1, sampling_interval_ms=20):
    experiment = ExperimentConfig(
        experiment_name="test",
        task=_TASK_NAME_CONSTANT_RATE,
        dataset=None,
        models=["dummy"],
        devices=["cpu"],
        precisions=["fp32"],
        batch_sizes=[1],
        phases=["inference"],
    )
    run_spec = experiment.expand_matrix()[0]
    run_spec.warmup.iterations = warmup_iterations
    run_spec.inference.num_iterations = num_iterations
    run_spec.power_meter.sampling_interval_ms = sampling_interval_ms
    run_spec.power_meter.cpu_sources = ["fake_slow_start_source"]
    run_spec.power_meter.smart_plug.enabled = False
    return run_spec


def test_retry_loop_accumulates_samples_across_attempts_despite_slow_starting_source(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "build_power_samplers",
        lambda device, power_meter: [_SlowStartFakeSampler(startup_delay_s=0.6)],
    )

    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())
    run_spec = _make_high_throughput_run_spec(num_iterations=50, warmup_iterations=1, sampling_interval_ms=20)

    outcome = runner._run_single(run_spec, device_override="cpu")

    assert outcome.status == "completed"
    assert len(repo.summaries) == 1
    summary = repo.summaries[0]

    assert summary.power_samples_count is not None
    assert summary.power_samples_count >= _MIN_POWER_SAMPLES_THRESHOLD, (
        f"power_samples_count utknal na {summary.power_samples_count} mimo dostepnego "
        f"zrodla i wielokrotnych prob podniesienia num_iterations (finalnie "
        f"{run_spec.inference.num_iterations}) - retry loop nie skumulowal probek "
        f"poprawnie miedzy probami"
    )
    assert summary.avg_power_watts is not None
    assert summary.avg_power_watts == pytest.approx(10.0)


# ============================================================
# Problem 4: dla batch_size=32/64 (wysoka przepustowosc) retry loop poprawnie dochodzil do
# TWARDEGO LIMITU "20x wartosci z configu" (np. 50 -> 1000 iteracji), ale przy tak duzym
# batchu 1000 iteracji zajmowalo tylko ~1.6-2.7s - KROCEJ niz jednorazowy narzut startowy
# codecarbon (~1.4-3s zanim odda pierwsza probke) - limit byl wiec osiagany PRZED tym, jak
# sampler zdazyl w ogole zaczac probkowac, niezaleznie od retry loop. Dwie poprawki:
# 1) limit czasowy (_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S) zamiast mnoznika iteracji,
# 2) dlugozyjacy sampler wspoldzielony miedzy przebiegami (ExperimentRunner::
#    _get_persistent_sampling_thread) - narzut startowy placi sie RAZ na cala serie, nie
#    przy kazdym przebiegu.
# ============================================================


def test_reaches_threshold_via_time_cap_even_when_iteration_multiple_of_config_is_far_exceeded(monkeypatch):
    # per_iter_s < 2ms ("bardzo wysoka przepustowosc" ze zgloszenia) + zrodlo z narzutem
    # startowym (3.0s) WIEKSZYM niz 20x*configured*per_iter_s (stary limit: 50*20*0.0018=
    # 1.8s < 3.0s) - stary limit nigdy nie pozwolilby dotrwac do momentu, w ktorym sampler
    # zaczyna faktycznie probkowac. Nowy limit czasowy (20s) ma na to duzo miejsca.
    per_iter_s = 0.0018
    _ConstantRateTask.PER_ITER_S = per_iter_s
    try:
        monkeypatch.setattr(
            orchestrator_module,
            "build_power_samplers",
            lambda device, power_meter: [_SlowStartFakeSampler(startup_delay_s=3.0)],
        )

        repo = _FakeRepository()
        runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())
        run_spec = _make_high_throughput_run_spec(num_iterations=50, warmup_iterations=1, sampling_interval_ms=20)

        outcome = runner._run_single(run_spec, device_override="cpu")

        assert outcome.status == "completed"
        summary = repo.summaries[0]
        assert summary.power_samples_count is not None
        assert summary.power_samples_count >= _MIN_POWER_SAMPLES_THRESHOLD, (
            f"power_samples_count={summary.power_samples_count} ponizej progu "
            f"{_MIN_POWER_SAMPLES_THRESHOLD} - limit czasowy powinien byl pozwolic "
            f"przetrwac jednorazowy narzut startowy zrodla (3.0s), nawet gdy stary limit "
            f"'20x configu' (1.8s) by na to nie pozwolil"
        )
        # Potwierdza, ze faktycznie wymagalo to iteracji DUZO powyzej "20x configu" (1000) -
        # inaczej test nie odroznialby starego zachowania od naprawionego.
        assert run_spec.inference.num_iterations > 50 * 20
    finally:
        _ConstantRateTask.PER_ITER_S = 0.0002  # przywroc domyslna wartosc dla innych testow


def test_persistent_sampler_pays_startup_cost_only_once_across_multiple_runs(monkeypatch):
    # "Wazniejsza, fundamentalna poprawka" ze zgloszenia: dlugozyjacy sampler jest tworzony
    # RAZ per ExperimentRunner/device_type, nie per-przebieg - drugi przebieg tego samego
    # typu urzadzenia w tej samej serii NIE placi ponownie jednorazowego narzutu
    # startowego, wiec osiaga prog probek bez czekania na "rozgrzanie" od zera.
    sampler = _SlowStartFakeSampler(startup_delay_s=2.0)
    monkeypatch.setattr(orchestrator_module, "build_power_samplers", lambda device, power_meter: [sampler])

    repo = _FakeRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())

    run_spec_1 = _make_high_throughput_run_spec(num_iterations=50, warmup_iterations=1, sampling_interval_ms=20)
    runner._run_single(run_spec_1, device_override="cpu")

    assert sampler._first_call_done is True, "pierwszy przebieg musial zdazyc 'rozgrzac' sampler"
    assert len(runner._persistent_power_threads) == 1, "sampler musi byc utworzony RAZ, nie per-przebieg"

    # Drugi przebieg: gdyby sampler byl tworzony od nowa (jak przed poprawka), musialby
    # znowu zaplacic 0.6s narzutu startowego, zanim cokolwiek by zebral. Mierzymy realny
    # czas trwania calego _run_single, zeby to bezposrednio potwierdzic/zaprzeczyc.
    run_spec_2 = _make_high_throughput_run_spec(num_iterations=50, warmup_iterations=1, sampling_interval_ms=20)
    t0 = time.perf_counter()
    outcome_2 = runner._run_single(run_spec_2, device_override="cpu")
    wall_clock_run_2_s = time.perf_counter() - t0

    assert outcome_2.status == "completed"
    summary_2 = repo.summaries[-1]
    assert summary_2.power_samples_count is not None
    assert summary_2.power_samples_count >= _MIN_POWER_SAMPLES_THRESHOLD

    # Margines: minimalny czas na zebranie progu probek (20 * 20ms = 0.4s) + narzut
    # zadania/retry loop pod obciazeniem systemu (rownolegle testy) - ale wyrazne
    # ODTWORZENIE narzutu startowego (2.0s) byloby od razu widoczne jako >=2.0s, wiec
    # margines 1.5s < 2.0s wciaz jednoznacznie odroznia "reuzyto sampler" od "zaplacono
    # narzut ponownie", bez bycia krucha wyscigowka ze stoperem.
    assert wall_clock_run_2_s < 1.5, (
        f"Drugi przebieg trwal {wall_clock_run_2_s:.3f}s - powinien byc znaczaco krotszy "
        f"niz narzut startowy zrodla ({sampler._startup_delay_s}s), skoro sampler byl juz "
        f"'rozgrzany' z pierwszego przebiegu. Tak dlugi czas sugeruje, ze narzut zostal "
        f"zaplacony ponownie (sampler zostal utworzony od nowa zamiast reuzyty)."
    )

    runner._close_persistent_sampling_threads()
    assert runner._persistent_power_threads == {}
