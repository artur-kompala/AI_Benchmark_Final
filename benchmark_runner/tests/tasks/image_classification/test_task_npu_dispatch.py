"""Testy dispatchu ImageClassificationTask wg device.backend - w szczegolnosci ze
faza 'train' na NPU (backend != 'pytorch') konczy sie czytelnym bledem zamiast proby
wykonania (NPU w tym ekosystemie wspiera tylko inferencje)."""

from __future__ import annotations

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.image_classification import onnx_export
from benchmark_runner.tasks.image_classification.onnx_export import ReferenceCheckpointNotFoundError
from benchmark_runner.tasks.image_classification.task import ImageClassificationTask
from benchmark_runner.utils.system_info import SystemInfo


def _make_run_context(backend: str, device_type: str) -> RunContext:
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


def test_run_train_epoch_raises_on_openvino_backend():
    run_context = _make_run_context(backend="openvino", device_type="npu_openvino")
    task = ImageClassificationTask(run_context)

    with pytest.raises(NotImplementedError, match="train"):
        task.run_train_epoch()


def test_run_train_epoch_raises_on_onnxruntime_backend():
    run_context = _make_run_context(backend="onnxruntime", device_type="npu_directml")
    task = ImageClassificationTask(run_context)

    with pytest.raises(NotImplementedError, match="train"):
        task.run_train_epoch()


def test_prepare_raises_on_unsupported_backend():
    run_context = _make_run_context(backend="llama_cpp", device_type="cpu")
    task = ImageClassificationTask(run_context)

    with pytest.raises(NotImplementedError):
        task.prepare()


def test_prepare_on_openvino_backend_raises_when_reference_checkpoint_missing(tmp_path, monkeypatch):
    # Bug: sciezka OpenVINO/ONNX Runtime (NPU/Intel GPU) eksportowala model o swiezo
    # zainicjalizowanych, losowych wagach do ONNX - accuracy mierzone pozniej bylo na
    # poziomie przypadku. prepare() musi odmowic (fail fast, PRZED probą eksportu do ONNX
    # czy kompilacji OpenVINO) zamiast po cichu kontynuowac z losowa inicjalizacja.
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "no_checkpoints_here")
    run_context = _make_run_context(backend="openvino", device_type="npu_openvino")
    task = ImageClassificationTask(run_context)

    with pytest.raises(ReferenceCheckpointNotFoundError, match="mobilenet_v3"):
        task.prepare()


def test_prepare_on_onnxruntime_backend_raises_when_reference_checkpoint_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "no_checkpoints_here")
    run_context = _make_run_context(backend="onnxruntime", device_type="npu_directml")
    task = ImageClassificationTask(run_context)

    with pytest.raises(ReferenceCheckpointNotFoundError, match="mobilenet_v3"):
        task.prepare()
