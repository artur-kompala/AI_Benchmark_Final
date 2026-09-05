"""Testy sanity-check accuracy PRZED zapisem do Supabase (core/orchestrator.py::
_check_accuracy_sanity) - accuracy bliskie poziomowi przypadku (np. ~10% dla CIFAR-10)
zdradza model o niewczytanych/losowych wagach, typowo brakujacy checkpoint referencyjny
na sciezce OpenVINO/ONNX Runtime (patrz tasks/image_classification/onnx_export.py).
_run_single rzuca ten wyjatek WEWNATRZ swojego try/except, wiec przebieg konczy sie
normalnie jako status='failed' zamiast 'completed' - nie przerywa reszty serii, ale
tez nie zanieczyszcza analizy podejrzanym wynikiem."""

from __future__ import annotations

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_task import TaskStepResult
from benchmark_runner.core.orchestrator import ImplausibleAccuracyError, _check_accuracy_sanity


def _make_run_spec(task: str = "image_classification", phase: str = "inference"):
    experiment = ExperimentConfig(
        experiment_name="test",
        task=task,
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[8],
        phases=[phase],
    )
    return experiment.expand_matrix()[0]


def test_raises_when_accuracy_near_chance_level_for_image_classification():
    run_spec = _make_run_spec()
    result = TaskStepResult(samples_processed=1000, accuracy=0.11)

    with pytest.raises(ImplausibleAccuracyError, match="0.3"):
        _check_accuracy_sanity(run_spec, result)


def test_does_not_raise_when_accuracy_above_threshold():
    run_spec = _make_run_spec()
    result = TaskStepResult(samples_processed=1000, accuracy=0.62)

    _check_accuracy_sanity(run_spec, result)  # no raise


def test_boundary_accuracy_exactly_at_threshold_does_not_raise():
    run_spec = _make_run_spec()
    result = TaskStepResult(samples_processed=1000, accuracy=0.3)

    _check_accuracy_sanity(run_spec, result)  # no raise - >= threshold is plausible


def test_does_not_raise_when_accuracy_is_none():
    run_spec = _make_run_spec()
    result = TaskStepResult(samples_processed=1000, accuracy=None)

    _check_accuracy_sanity(run_spec, result)  # brak accuracy (np. llm_inference) - nic do sprawdzenia


def test_does_not_raise_for_train_phase_even_with_low_accuracy():
    # W trakcie treningu niska accuracy (np. pierwsza epoka) jest oczekiwana/normalna -
    # sanity check dotyczy wylacznie fazy inference.
    run_spec = _make_run_spec(phase="train")
    result = TaskStepResult(samples_processed=1000, accuracy=0.05)

    _check_accuracy_sanity(run_spec, result)  # no raise


def test_does_not_raise_for_task_without_configured_threshold():
    # llm_inference nie ma progu skonfigurowanego w _MIN_PLAUSIBLE_ACCURACY_BY_TASK -
    # nawet gdyby result.accuracy bylo ustawione, nie ma tu sensownego progu do porownania.
    run_spec = _make_run_spec(task="llm_inference")
    result = TaskStepResult(samples_processed=1000, accuracy=0.05)

    _check_accuracy_sanity(run_spec, result)  # no raise


def test_error_message_includes_run_context_for_diagnosis():
    run_spec = _make_run_spec()
    result = TaskStepResult(samples_processed=1000, accuracy=0.1)

    with pytest.raises(ImplausibleAccuracyError) as exc_info:
        _check_accuracy_sanity(run_spec, result)

    message = str(exc_info.value)
    assert "mobilenet_v3" in message
    assert "fp32" in message
