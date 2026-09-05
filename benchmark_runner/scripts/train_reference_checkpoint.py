"""Trenuje raz i zapisuje checkpoint referencyjny (wytrenowane wagi) dla modeli
klasyfikacji obrazow, uzywany przez sciezke inference-only OpenVINO/ONNX Runtime
(NPU/Intel GPU - patrz tasks/image_classification/task.py::_prepare_onnx). Te urzadzenia
nigdy same nie traninguja (faza 'train' konczy sie tam NotImplementedError), wiec bez
tego checkpointu benchmark eksportowalby do ONNX model o swiezo zainicjalizowanych,
losowych wagach - zmierzona accuracy oscylowalaby wokol poziomu przypadku (~10% dla
CIFAR-10) niezaleznie od device/precision.

Uruchom RAZ na maszynie z CPU/GPU_AMD/GPU_NVIDIA (dowolne z nich - to zwykly trening
PyTorch, nie wymaga NPU/Intel GPU), potem skopiuj katalog data/checkpoints/ (albo po
prostu ten sam plik .pt) na kazda maszyne testowa uruchamiajaca serie na NPU/Intel GPU -
patrz tasks/image_classification/onnx_export.py (CHECKPOINT_DIR, REFERENCE_SEED).

Uzycie:
    python scripts/train_reference_checkpoint.py --model mobilenet_v3
    python scripts/train_reference_checkpoint.py --model resnet50 --epochs 20
    python scripts/train_reference_checkpoint.py --model all   # wszystkie z MODEL_REGISTRY

Wymaga aktywnego venv z zainstalowanymi zaleznosciami wspolnymi (torch/torchvision,
patrz requirements/requirements-common.txt).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

from benchmark_runner.tasks.image_classification.data import get_cifar10_dataloaders  # noqa: E402
from benchmark_runner.tasks.image_classification.models import MODEL_REGISTRY, build_model  # noqa: E402
from benchmark_runner.tasks.image_classification.onnx_export import (  # noqa: E402
    REFERENCE_SEED,
    checkpoint_path_for,
    save_reference_checkpoint,
)
from benchmark_runner.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Znaczaco wiecej niz warmup+inference potrzebuja pojedynczo mierzone przebiegi - to
# jest jednorazowy trening referencyjny, ma miec porzadna liczbe epok nad malym
# podzbiorem treningowym (_TRAIN_SUBSET_SIZE=5000 w data.py), zeby accuracy bylo
# wyraznie ponad prog sanity-check (>0.3, patrz core/orchestrator.py::_check_accuracy_sanity),
# nie tylko "troche lepiej niz losowo".
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)


def _train_one_model(model_name: str, epochs: int, batch_size: int, learning_rate: float, seed: int) -> float:
    logger.info("=== Trening referencyjny: model=%s, epochs=%d, batch_size=%d, seed=%d ===", model_name, epochs, batch_size, seed)
    _seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        train_loss = total_loss / total_samples if total_samples else float("nan")
        logger.info("[%s] epoka %d/%d - train_loss=%.4f (%.1fs)", model_name, epoch, epochs, train_loss, time.perf_counter() - t0)

    accuracy = _evaluate(model, test_loader, device)
    logger.info("[%s] accuracy na test set po treningu: %.4f", model_name, accuracy)

    save_path = save_reference_checkpoint(model.to("cpu"), model_name)
    logger.info("[%s] checkpoint zapisany: %s", model_name, save_path)
    return accuracy


def _evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += int((outputs.argmax(dim=1) == targets).sum().item())
            total += inputs.size(0)
    return correct / total if total else 0.0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_reference_checkpoint",
        description="Trenuje i zapisuje checkpoint referencyjny CIFAR-10 dla sciezki OpenVINO/ONNX Runtime (NPU/Intel GPU)",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=[*sorted(MODEL_REGISTRY), "all"],
        help="Ktory model wytrenowac (domyslnie: all - wszystkie z MODEL_REGISTRY)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--seed",
        type=int,
        default=REFERENCE_SEED,
        help=f"Domyslnie {REFERENCE_SEED} (onnx_export.REFERENCE_SEED) - zmiana wymaga tez zmiany "
        "REFERENCE_SEED w kodzie, inaczej checkpoint trafi pod inna nazwe pliku niz ta, ktorej "
        "szuka load_reference_checkpoint()",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nadpisz istniejacy checkpoint bez pytania (domyslnie skrypt pyta, jesli plik juz istnieje)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_arg_parser().parse_args(argv)

    model_names = sorted(MODEL_REGISTRY) if args.model == "all" else [args.model]

    for model_name in model_names:
        existing = checkpoint_path_for(model_name)
        if existing.exists() and not args.force:
            answer = input(f"Checkpoint {existing} juz istnieje - nadpisac? [y/N] ").strip().lower()
            if answer != "y":
                logger.info("Pomijam '%s' - checkpoint pozostaje bez zmian", model_name)
                continue

        accuracy = _train_one_model(
            model_name, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed
        )
        if accuracy < 0.3:
            logger.warning(
                "[%s] accuracy=%.4f jest ponizej progu sanity-check (0.3, patrz "
                "core/orchestrator.py::_MIN_PLAUSIBLE_ACCURACY_BY_TASK) - rozwaz zwiekszenie "
                "--epochs, ten checkpoint moze byc odrzucany przez benchmark",
                model_name,
                accuracy,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
