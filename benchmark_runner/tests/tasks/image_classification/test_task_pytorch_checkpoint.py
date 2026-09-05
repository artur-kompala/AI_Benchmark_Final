"""Testy wymogu checkpointu referencyjnego na sciezce pytorch (CPU/CUDA/ROCm) dla fazy
inference. Ta sciezka wczesniej ZAWSZE budowala model od losowej inicjalizacji (tak samo
jak sciezka OpenVINO/ONNX Runtime przed poprzednia poprawka) - accuracy z inferencji na
CPU/GPU_NVIDIA/GPU_AMD bylaby wiec rownie bez sensu. Faza 'train' celowo NADAL zaczyna od
losowych wag (mierzymy dynamike/koszt treningu, nie koncowa dokladnosc), wiec tam
checkpoint nie jest wymagany."""

from __future__ import annotations

import pytest

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.image_classification import onnx_export
from benchmark_runner.tasks.image_classification.onnx_export import ReferenceCheckpointNotFoundError
from benchmark_runner.tasks.image_classification.task import ImageClassificationTask
from benchmark_runner.utils.system_info import SystemInfo


def _make_run_context(phase: str, device_type: str = "cpu") -> RunContext:
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[8],
        phases=[phase],
    )
    run_spec = experiment.expand_matrix()[0]
    device = DeviceProfile(device_type=device_type, label="Test Device", vendor=None, torch_device_str="cpu", backend="pytorch")
    system_info = SystemInfo(machine_name="test-machine", os_name="Linux", os_version="1", cpu_model="Test CPU", total_ram_gb=16.0)
    return RunContext(run_spec=run_spec, device=device, system_info=system_info)


def test_prepare_inference_on_pytorch_backend_raises_when_checkpoint_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "no_checkpoints_here")
    run_context = _make_run_context(phase="inference")
    task = ImageClassificationTask(run_context)

    with pytest.raises(ReferenceCheckpointNotFoundError, match="mobilenet_v3"):
        task.prepare()


def test_prepare_train_on_pytorch_backend_does_not_require_checkpoint(tmp_path, monkeypatch):
    # Faza 'train' zaczyna od losowej inicjalizacji celowo - nie powinna nawet probowac
    # wczytac checkpointu, wiec musi dzialac mimo pustego CHECKPOINT_DIR.
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "no_checkpoints_here")
    run_context = _make_run_context(phase="train")
    task = ImageClassificationTask(run_context)

    task.prepare()  # nie powinno rzucic

    assert task._model is not None
    assert task._reference_accuracy is None
