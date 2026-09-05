"""Testy dla logiki powtorzen (Poprawka 2): _filter_unsupported_combos (int8+train
pomijane przed wykonaniem) i _finalize_repetition_group (mean/std/coefficient of
variation po wszystkich powtorzeniach jednej kombinacji, zapisane do run_stats_summary)."""

from __future__ import annotations

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.orchestrator import ExperimentRunner, _filter_unsupported_combos, _RunOutcome


class _FakeStatsRepository:
    def __init__(self):
        self.stats = []

    def upsert_run_stats_summary(self, stats):
        self.stats.append(stats)


class _FakeLocalBuffer:
    def flush_pending(self, repository):
        return 0


def _make_group_meta():
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["cpu"],
        precisions=["fp32"],
        batch_sizes=[8],
        phases=["inference"],
        repetitions=1,
    )
    return experiment.expand_matrix()[0]


def test_filter_unsupported_combos_drops_int8_train_keeps_rest():
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["cpu"],
        precisions=["fp32", "int8"],
        batch_sizes=[8],
        phases=["train", "inference"],
        repetitions=1,
    )
    run_specs = experiment.expand_matrix()
    # 2 precisions x 1 batch x 2 phases = 4 kombinacje przed filtrowaniem
    assert len(run_specs) == 4

    filtered = _filter_unsupported_combos(run_specs)
    combos = {(r.precision, r.phase) for r in filtered}
    assert ("int8", "train") not in combos
    assert ("int8", "inference") in combos
    assert ("fp32", "train") in combos
    assert ("fp32", "inference") in combos
    assert len(filtered) == 3


def test_filter_unsupported_combos_drops_batch_size_1_train_for_image_classification():
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["cpu"],
        precisions=["fp32"],
        batch_sizes=[1, 32],
        phases=["train", "inference"],
        repetitions=1,
    )
    run_specs = experiment.expand_matrix()
    assert len(run_specs) == 4

    filtered = _filter_unsupported_combos(run_specs)
    combos = {(r.batch_size, r.phase) for r in filtered}
    # BatchNorm (MobileNetV3/ResNet-50) nie wspiera treningu z batch_size=1.
    assert (1, "train") not in combos
    assert (1, "inference") in combos  # inferencja z batch=1 dziala normalnie (running stats)
    assert (32, "train") in combos
    assert (32, "inference") in combos
    assert len(filtered) == 3


def test_filter_unsupported_combos_keeps_batch_size_1_train_for_nlp_sentiment():
    experiment = ExperimentConfig(
        experiment_name="test",
        task="nlp_sentiment",
        dataset="imdb",
        models=["distilbert-base-uncased"],
        devices=["cpu"],
        precisions=["fp32"],
        batch_sizes=[1],
        phases=["train"],
        repetitions=1,
    )
    run_specs = experiment.expand_matrix()
    filtered = _filter_unsupported_combos(run_specs)
    # DistilBERT uzywa LayerNorm (nie BatchNorm) - batch_size=1 w treningu dziala normalnie,
    # filtr jest zawezony do task=='image_classification' i nie powinien tego dotknac.
    assert len(filtered) == 1


def test_finalize_repetition_group_computes_mean_and_stdev():
    repo = _FakeStatsRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())
    group_meta = _make_group_meta()

    outcomes = [
        _RunOutcome(status="completed", device_id="dev-1", duration_s=10.0, energy_joules_software=100.0),
        _RunOutcome(status="completed", device_id="dev-1", duration_s=12.0, energy_joules_software=110.0),
        _RunOutcome(status="completed", device_id="dev-1", duration_s=14.0, energy_joules_software=120.0),
    ]

    runner._finalize_repetition_group(group_meta, outcomes)

    assert len(repo.stats) == 1
    stats = repo.stats[0]
    assert stats.n_repetitions == 3
    assert stats.mean_duration_s == pytest.approx(12.0)
    assert stats.std_duration_s == pytest.approx(2.0)  # sample stdev of [10, 12, 14]
    assert stats.mean_energy_joules == pytest.approx(110.0)
    assert stats.coefficient_of_variation_energy == pytest.approx(stats.std_energy_joules / 110.0)
    assert stats.device_id == "dev-1"


def test_finalize_repetition_group_excludes_failed_runs():
    repo = _FakeStatsRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())
    group_meta = _make_group_meta()

    outcomes = [
        _RunOutcome(status="completed", device_id="dev-1", duration_s=10.0, energy_joules_software=100.0),
        _RunOutcome(status="failed", device_id="dev-1", duration_s=None, energy_joules_software=None),
    ]

    runner._finalize_repetition_group(group_meta, outcomes)

    assert len(repo.stats) == 1
    assert repo.stats[0].n_repetitions == 1
    assert repo.stats[0].std_duration_s == 0.0  # jedno uzyteczne powtorzenie -> std=0, nie None


def test_finalize_repetition_group_skips_when_no_successful_runs():
    repo = _FakeStatsRepository()
    runner = ExperimentRunner(repository=repo, local_buffer=_FakeLocalBuffer())
    group_meta = _make_group_meta()

    outcomes = [
        _RunOutcome(status="failed", device_id="dev-1", duration_s=None, energy_joules_software=None),
        _RunOutcome(status="completed", device_id=None, duration_s=5.0, energy_joules_software=50.0),
    ]

    runner._finalize_repetition_group(group_meta, outcomes)

    assert repo.stats == []
