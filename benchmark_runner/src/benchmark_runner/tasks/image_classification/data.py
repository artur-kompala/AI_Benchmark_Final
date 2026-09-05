"""DataLoadery CIFAR-10 (torchvision), pobierane automatycznie przy pierwszym uruchomieniu."""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# .../benchmark_runner/data/cifar10 (poza src/, w .gitignore)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "cifar10"

# Benchmark mierzy czas/moc/throughput, nie dokladnosc modelu (analogicznie do IMDB w
# tasks/nlp_sentiment/data.py), wiec ograniczamy PELNY CIFAR-10 (50k/10k) do podzbioru dla
# rozsadnego czasu przebiegu na zroznicowanej flocie maszyn testowych (w tym slabszych
# laptopow) - przy num_epochs>1 x wielu precyzjach/batch_sizes/repetitions pelny zbior
# wielokrotnie wydluzalby serie bez dodatkowej wartosci dla pomiaru czasu/energii/przepustowosci.
_TRAIN_SUBSET_SIZE = 5000
_TEST_SUBSET_SIZE = 1000
_SUBSET_SEED = 42


def _subset(dataset, size: int, seed: int):
    size = min(size, len(dataset))
    indices = random.Random(seed).sample(range(len(dataset)), k=size)
    return Subset(dataset, indices)


def get_cifar10_dataloaders(
    batch_size: int,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    num_workers: int = 2,
    drop_last_test: bool = False,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )

    train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)

    train_set = _subset(train_set, _TRAIN_SUBSET_SIZE, _SUBSET_SEED)
    test_set = _subset(test_set, _TEST_SUBSET_SIZE, _SUBSET_SEED + 1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # drop_last_test=True dla sciezek inferencji o statycznym ksztalcie wejscia
    # (OpenVINO/ONNX Runtime, patrz tasks/image_classification/task.py::_prepare_onnx) -
    # bez tego ostatni, niepelny batch wywala inferencje ("tensor size is not equal to
    # model"). Sciezka pytorch (drop_last_test=False, domyslnie) toleruje ragged batch.
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last_test
    )
    return train_loader, test_loader
