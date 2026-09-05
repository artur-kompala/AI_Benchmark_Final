"""DataLoadery IMDB sentiment (Hugging Face datasets), tokenizacja przez DistilBERT tokenizer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256

# Benchmark mierzy czas/moc/throughput, nie dokladnosc modelu, wiec ograniczamy IMDB
# (25k/25k) do malego podzbioru dla rozsadnego czasu przebiegu - zgodnie z oryginalnym
# wymaganiem "maly zbior tekstowy".
_TRAIN_SUBSET_SIZE = 2000
_TEST_SUBSET_SIZE = 500


class _TokenizedImdbDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        self._encodings = encodings
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self._encodings.items()}
        item["labels"] = self._labels[idx]
        return item


def get_imdb_dataloaders(batch_size: int, cache_dir: str | Path | None = None) -> tuple[DataLoader, DataLoader]:
    from datasets import load_dataset
    from transformers import DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(cache_dir) if cache_dir else None)
    train_subset = dataset["train"].shuffle(seed=42).select(range(_TRAIN_SUBSET_SIZE))
    test_subset = dataset["test"].shuffle(seed=42).select(range(_TEST_SUBSET_SIZE))

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)

    train_subset = train_subset.map(tokenize, batched=True)
    test_subset = test_subset.map(tokenize, batched=True)

    train_loader = DataLoader(
        _TokenizedImdbDataset(
            {
                "input_ids": torch.tensor(train_subset["input_ids"]),
                "attention_mask": torch.tensor(train_subset["attention_mask"]),
            },
            torch.tensor(train_subset["label"]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        _TokenizedImdbDataset(
            {
                "input_ids": torch.tensor(test_subset["input_ids"]),
                "attention_mask": torch.tensor(test_subset["attention_mask"]),
            },
            torch.tensor(test_subset["label"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
