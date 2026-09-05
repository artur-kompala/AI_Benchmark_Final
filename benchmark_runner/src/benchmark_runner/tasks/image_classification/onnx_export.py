"""Eksport modelu PyTorch do ONNX - uzywany przez sciezke NPU/Intel GPU (OpenVINO/ONNX
Runtime DirectML).

Kazdy przebieg benchmarku ma ustalony batch_size z configu, wiec eksportujemy ze
stalym ksztaltem wejscia (batch_size, *input_shape) - NPU/OpenVINO wymaga
skompilowanego grafu i nie potrzebujemy tu dynamicznego rozmiaru batcha.

Ta sciezka (OpenVINO/ONNX Runtime na NPU/Intel GPU) NIGDY sama nie trenuje modelu
(patrz task.py::run_train_epoch - faza 'train' na tych urzadzeniach konczy sie
NotImplementedError) - potrzebuje wiec wczytac wagi wytrenowane raz gdzie indziej
(CPU/GPU_AMD/GPU_NVIDIA, seed=REFERENCE_SEED, patrz scripts/train_reference_checkpoint.py).
Bez tego eksportowany jest model o swiezo zainicjalizowanych (losowych) wagach, a
zmierzona accuracy oscyluje wokol poziomu przypadku (np. ~10% dla CIFAR-10, 10 klas)
niezaleznie od device/precision - load_reference_checkpoint() ponizej rzuca czytelny
blad zamiast po cichu kontynuowac z takim modelem.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# .../benchmark_runner/data/checkpoints (poza src/, tak jak data/cifar10 w data.py) -
# generowane lokalnie przez scripts/train_reference_checkpoint.py, nie w repo.
CHECKPOINT_DIR = Path(__file__).resolve().parents[4] / "data" / "checkpoints"

# Seed uzyty do wytrenowania checkpointu referencyjnego - ten sam co domyslny
# ExperimentConfig.seed (config/schema.py), zeby trening referencyjny byl odtwarzalny
# tymi samymi narzedziami co reszta benchmarku (utils/seed przez _seed_everything).
REFERENCE_SEED = 42


class ReferenceCheckpointNotFoundError(RuntimeError):
    """Brak checkpointu referencyjnego pod oczekiwana sciezka - sciezka inference-only
    (OpenVINO/ONNX Runtime) MUSI odmowic kontynuacji z losowymi wagami zamiast po cichu
    mierzyc accuracy bez sensu (patrz modul docstring wyzej)."""


def checkpoint_path_for(model_name: str, dataset: str = "cifar10") -> Path:
    return CHECKPOINT_DIR / f"{model_name}_{dataset}_seed{REFERENCE_SEED}.pt"


def load_reference_checkpoint(model: nn.Module, model_name: str, dataset: str = "cifar10") -> None:
    """Wczytuje wytrenowane wagi do `model` (in-place, przez load_state_dict). Rzuca
    ReferenceCheckpointNotFoundError z czytelna instrukcja naprawy, jesli checkpoint nie
    istnieje - wywolujacy (task.py::_prepare_onnx) NIE powinien lapac tego wyjatku i
    kontynuowac eksportu z losowa inicjalizacja."""
    path = checkpoint_path_for(model_name, dataset)
    if not path.exists():
        raise ReferenceCheckpointNotFoundError(
            f"Brak checkpointu referencyjnego dla modelu '{model_name}' pod {path}. Sciezka "
            "inference-only (OpenVINO/ONNX Runtime na NPU/Intel GPU) wymaga wytrenowanych wag, "
            "zeby zmierzona accuracy miala sens - losowa inicjalizacja daje accuracy na poziomie "
            "przypadku. Wygeneruj checkpoint (raz, na maszynie z CPU/GPU_AMD/GPU_NVIDIA):\n"
            f"    python scripts/train_reference_checkpoint.py --model {model_name}"
        )
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    logger.info("Wczytano checkpoint referencyjny dla modelu '%s' z %s", model_name, path)


def save_reference_checkpoint(model: nn.Module, model_name: str, dataset: str = "cifar10") -> Path:
    """Zapisuje state_dict modelu jako checkpoint referencyjny - wolane wylacznie przez
    scripts/train_reference_checkpoint.py po zakonczonym treningu."""
    path = checkpoint_path_for(model_name, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Zapisano checkpoint referencyjny dla modelu '%s' do %s", model_name, path)
    return path


def export_to_onnx(
    model: torch.nn.Module,
    batch_size: int,
    input_shape: tuple[int, ...],
    output_path: str | Path,
    opset_version: int = 18,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    dummy_input = torch.randn(batch_size, *input_shape)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            # dynamo=True (domyslne od torch 2.x) wymaga dodatkowej zaleznosci
            # 'onnxscript' - patrz requirements-npu.txt. Nowy eksporter jest aktywnie
            # rozwijany przez zespol PyTorch, legacy TorchScript exporter jest
            # zapowiedziany do usuniecia.
            dynamo=True,
            # verbose=False: domyslny (None) tryb dynamo drukuje na stdout m.in. znak
            # "✅", ktory wywala UnicodeEncodeError w standardowej konsoli Windows
            # (codepage cp1252) - jawnie wylaczamy te wewnetrzne logi eksportera.
            verbose=False,
        )
    finally:
        model.train(was_training)

    logger.info("Wyeksportowano model do ONNX: %s", output_path)
    return output_path
