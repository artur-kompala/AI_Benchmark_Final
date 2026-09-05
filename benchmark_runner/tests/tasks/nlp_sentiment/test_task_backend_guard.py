"""NlpSentimentTask obsluguje na razie tylko backend 'pytorch' - konstruktor powinien
odrzucic urzadzenia NPU (backend openvino/onnxruntime) od razu, czytelnym bledem."""

from __future__ import annotations

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.nlp_sentiment.task import NlpSentimentTask
from benchmark_runner.utils.system_info import SystemInfo


def _make_run_context(backend: str, device_type: str) -> RunContext:
    experiment = ExperimentConfig(
        experiment_name="test",
        task="nlp_sentiment",
        dataset="imdb",
        models=["distilbert-base-uncased"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[8],
        phases=["inference"],
    )
    run_spec = experiment.expand_matrix()[0]

    device = DeviceProfile(
        device_type=device_type,
        label="Test Device",
        vendor=None,
        torch_device_str=None,
        backend=backend,
    )
    system_info = SystemInfo(
        machine_name="test-machine",
        os_name="Windows",
        os_version="10",
        cpu_model="Test CPU",
        total_ram_gb=16.0,
    )
    return RunContext(run_spec=run_spec, device=device, system_info=system_info)


def test_init_raises_on_openvino_backend():
    run_context = _make_run_context(backend="openvino", device_type="npu_openvino")
    with pytest.raises(NotImplementedError, match="pytorch"):
        NlpSentimentTask(run_context)


def test_init_succeeds_on_pytorch_backend():
    run_context = _make_run_context(backend="pytorch", device_type="cpu")
    run_context.device = DeviceProfile(
        device_type="cpu", label="CPU", vendor=None, torch_device_str="cpu", backend="pytorch"
    )
    task = NlpSentimentTask(run_context)
    assert task is not None
