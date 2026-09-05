"""Iterator dla sciezek inferencji o statycznym ksztalcie wejscia (OpenVINO/ONNX Runtime).

Model skompilowany przez OpenVINO/ONNX Runtime (patrz tasks/image_classification/
onnx_export.py) ma STATYCZNY ksztalt wejscia ustalony na etapie eksportu (batch_size z
configu) - jakikolwiek batch o innym rozmiarze (typowo ostatni, niepelny batch z DataLoadera)
wywala inferencje bledem "tensor size is not equal to model". cycling_batches() gwarantuje,
ze kazdy zwrocony batch ma dokladnie loader.batch_size probek, a gdy dataset (po drop_last)
ma mniej pelnych batchy niz wymagana liczba iteracji, zawija go cyklicznie od poczatku.
"""

from __future__ import annotations

import logging
from typing import Iterator, TypeVar

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def cycling_batches(loader: DataLoader, n_batches: int) -> Iterator[_T]:
    """Zwraca dokladnie `n_batches` batchy z `loader`, zawijajac dataset cyklicznie od
    poczatku, jesli `loader` wyczerpie sie zanim dostarczy ich tyle. Wymaga `loader`
    skonfigurowanego z drop_last=True, zeby kazdy zwrocony batch mial staly rozmiar."""
    if not loader.drop_last:
        raise ValueError(
            "cycling_batches wymaga DataLoader skonfigurowanego z drop_last=True - w "
            "przeciwnym razie ostatni batch w kazdym przejsciu moze miec rozmiar inny "
            "niz batch_size, co wywala inferencje na modelu o statycznym ksztalcie wejscia."
        )

    n_full_batches = len(loader)
    if n_full_batches == 0:
        raise ValueError(
            f"DataLoader nie ma ani jednego pelnego batcha o rozmiarze {loader.batch_size} "
            f"(dataset ma {len(loader.dataset)} probek) - zmniejsz batch_size w configu albo "
            "zwieksz rozmiar datasetu."
        )

    produced = 0
    pass_idx = 0
    while produced < n_batches:
        pass_idx += 1
        if pass_idx == 2:
            logger.info(
                "Dataloader wyczerpany po %d pelnych batchach (batch_size=%s) - zawijam "
                "dataset cyklicznie, zeby dostarczyc wymagane %d iteracji; czesc probek w "
                "tym przebiegu zostanie powtorzona.",
                n_full_batches,
                loader.batch_size,
                n_batches,
            )
        for batch in loader:
            if produced >= n_batches:
                return
            yield batch
            produced += 1
