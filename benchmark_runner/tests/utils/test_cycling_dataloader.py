"""Testy cycling_batches (utils/cycling_dataloader.py) - gwarancja stalego rozmiaru batcha
dla sciezek inferencji OpenVINO/ONNX Runtime o statycznym ksztalcie wejscia (Blad 1:
"tensor size is not equal to model" przy niepelnym ostatnim batchu, patrz
tasks/image_classification/task.py::_run_inference_onnx)."""

from __future__ import annotations

import logging

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark_runner.utils.cycling_dataloader import cycling_batches


def _make_loader(dataset_size: int, batch_size: int, drop_last: bool = True) -> DataLoader:
    x = torch.arange(dataset_size).float().unsqueeze(1)
    y = torch.arange(dataset_size)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=drop_last)


@pytest.mark.parametrize(
    "dataset_size,batch_size,n_batches",
    [
        (1000, 32, 31),   # dokladnie tyle ile pelnych batchy jest dostepnych - brak zawijania
        (1000, 32, 5 + 50),  # warmup(5) + inference(50), mniej niz 31 pelnych batchy - bez zawijania
        (1000, 64, 5 + 50),  # tylko 15 pelnych batchy, potrzeba 55 - MUSI zawinac dataset
        (100, 32, 3),      # dataset niepodzielny przez batch_size (3 pelne batche, reszta odrzucona)
        (17, 8, 10),       # bardzo maly dataset, wielokrotne zawijanie w jednym wywolaniu
        (1, 1, 7),         # skrajny przypadek - dataset o rozmiarze 1
    ],
)
def test_cycling_batches_never_yields_wrong_batch_size(dataset_size, batch_size, n_batches):
    loader = _make_loader(dataset_size, batch_size)
    batches = list(cycling_batches(loader, n_batches))

    assert len(batches) == n_batches
    for inputs, targets in batches:
        assert inputs.shape[0] == batch_size
        assert targets.shape[0] == batch_size


def test_cycling_batches_logs_info_when_wrapping_occurs(caplog):
    loader = _make_loader(dataset_size=1000, batch_size=64)  # 15 pelnych batchy
    with caplog.at_level(logging.INFO, logger="benchmark_runner.utils.cycling_dataloader"):
        list(cycling_batches(loader, n_batches=55))  # 55 > 15 -> musi zawinac

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("zawijam" in r.message for r in info_records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_cycling_batches_does_not_log_when_no_wrap_needed(caplog):
    loader = _make_loader(dataset_size=1000, batch_size=32)  # 31 pelnych batchy
    with caplog.at_level(logging.INFO, logger="benchmark_runner.utils.cycling_dataloader"):
        list(cycling_batches(loader, n_batches=10))  # 10 <= 31 -> bez zawijania

    assert len(caplog.records) == 0


def test_cycling_batches_requires_drop_last():
    loader = _make_loader(dataset_size=100, batch_size=32, drop_last=False)
    with pytest.raises(ValueError, match="drop_last=True"):
        list(cycling_batches(loader, n_batches=5))


def test_cycling_batches_raises_on_dataset_smaller_than_batch_size():
    loader = _make_loader(dataset_size=5, batch_size=32)  # 0 pelnych batchy
    with pytest.raises(ValueError, match="ani jednego pelnego batcha"):
        list(cycling_batches(loader, n_batches=5))
