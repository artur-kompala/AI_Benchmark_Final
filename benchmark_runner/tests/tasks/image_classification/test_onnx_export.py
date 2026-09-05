from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from benchmark_runner.tasks.image_classification import onnx_export
from benchmark_runner.tasks.image_classification.onnx_export import (
    ReferenceCheckpointNotFoundError,
    checkpoint_path_for,
    export_to_onnx,
    load_reference_checkpoint,
    save_reference_checkpoint,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_export_to_onnx_creates_file(tmp_path):
    model = _TinyModel()
    output_path = tmp_path / "tiny_model.onnx"

    result_path = export_to_onnx(model, batch_size=4, input_shape=(3, 32, 32), output_path=output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_to_onnx_restores_training_mode(tmp_path):
    model = _TinyModel()
    model.train()
    export_to_onnx(model, batch_size=2, input_shape=(3, 32, 32), output_path=tmp_path / "m.onnx")
    assert model.training is True


def test_export_to_onnx_creates_parent_directories(tmp_path):
    model = _TinyModel()
    output_path = tmp_path / "nested" / "dir" / "model.onnx"

    export_to_onnx(model, batch_size=1, input_shape=(3, 32, 32), output_path=output_path)

    assert output_path.exists()


# ============================================================
# Checkpoint referencyjny (Blad: sciezka OpenVINO/ONNX Runtime na NPU/Intel GPU
# eksportowala model o swiezo zainicjalizowanych, losowych wagach - accuracy na
# poziomie przypadku niezaleznie od device/precision). load_reference_checkpoint()
# musi odmowic kontynuacji zamiast po cichu uzyc losowych wag.
# ============================================================


def test_load_reference_checkpoint_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    model = _TinyModel()

    with pytest.raises(ReferenceCheckpointNotFoundError, match="mobilenet_v3"):
        load_reference_checkpoint(model, "mobilenet_v3")


def test_load_reference_checkpoint_error_message_includes_fix_command(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    model = _TinyModel()

    with pytest.raises(ReferenceCheckpointNotFoundError, match="train_reference_checkpoint.py"):
        load_reference_checkpoint(model, "resnet50")


def test_save_then_load_reference_checkpoint_round_trips_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    trained = _TinyModel()
    with torch.no_grad():
        trained.fc.weight.fill_(0.1234)  # wartosc jednoznacznie rozna od domyslnej inicjalizacji

    saved_path = save_reference_checkpoint(trained, "tiny")
    assert saved_path.exists()
    assert saved_path == checkpoint_path_for("tiny")

    fresh = _TinyModel()
    assert not torch.allclose(fresh.fc.weight, trained.fc.weight)

    load_reference_checkpoint(fresh, "tiny")
    assert torch.allclose(fresh.fc.weight, trained.fc.weight)


def test_checkpoint_path_for_is_specific_to_model_and_dataset():
    path_a = checkpoint_path_for("mobilenet_v3", dataset="cifar10")
    path_b = checkpoint_path_for("resnet50", dataset="cifar10")
    assert path_a != path_b
    assert "mobilenet_v3" in path_a.name
    assert "resnet50" in path_b.name
