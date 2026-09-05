"""Szacowanie FLOPS modelu (thop/fvcore) na potrzeby metryki FLOPS/W.

Statyczne oszacowanie liczby operacji zmiennoprzecinkowych JEDNEGO przejscia forward
(batch=1) - wlasciwosc architektury modelu, niezalezna od precyzji/urzadzenia wykonania,
stad liczona raz w task.prepare() z modelu fp32 PRZED ewentualna kwantyzacja (patrz
tasks/image_classification/task.py, tasks/nlp_sentiment/task.py)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def estimate_flops(
    model: Any,
    input_shape: tuple[int, ...] | None = None,
    dummy_inputs: tuple[Any, ...] | None = None,
) -> float | None:
    """Szacuje liczbe FLOPS dla jednego przejscia forward. Podaj DOKLADNIE JEDNO z:
    - input_shape: ksztalt wejscia bez wymiaru batch (np. (3, 32, 32) dla CIFAR-10) -
      buduje domyslny dummy input torch.randn(1, *input_shape); wystarcza dla modeli
      przyjmujacych pojedynczy tensor float (CNN klasyfikacji obrazow).
    - dummy_inputs: gotowa krotka argumentow pozycyjnych do model.forward(*dummy_inputs)
      - wymagane dla modeli o innym niz pojedynczy float-tensor sygnaturze wejscia (np.
      transformery HuggingFace: (input_ids, attention_mask), oba LongTensor).
    Best-effort: probuje thop, potem fvcore; None gdy oba zawioda (nie przerywa
    benchmarku - flops_per_sample/flops_per_watt w wynikach zostaja wtedy NULL)."""
    if dummy_inputs is None:
        if input_shape is None:
            raise ValueError("Podaj input_shape lub dummy_inputs")
        import torch

        dummy_inputs = (torch.randn(1, *input_shape),)

    try:
        return _estimate_with_thop(model, dummy_inputs)
    except Exception:
        logger.debug("thop nie zadzialal przy szacowaniu FLOPS, probuje fvcore", exc_info=True)

    try:
        return _estimate_with_fvcore(model, dummy_inputs)
    except Exception:
        logger.warning("Nie udalo sie oszacowac FLOPS modelu (thop i fvcore zawiodly)", exc_info=True)
        return None


def _estimate_with_thop(model: Any, dummy_inputs: tuple[Any, ...]) -> float:
    from thop import profile

    was_training = model.training
    model.eval()
    try:
        macs, _params = profile(model, inputs=dummy_inputs, verbose=False)
    finally:
        model.train(was_training)
    return float(macs) * 2  # 1 MAC = 2 FLOPS


def _estimate_with_fvcore(model: Any, dummy_inputs: tuple[Any, ...]) -> float:
    from fvcore.nn import FlopCountAnalysis

    was_training = model.training
    model.eval()
    try:
        total_macs = FlopCountAnalysis(model, dummy_inputs).total()
    finally:
        model.train(was_training)
    return float(total_macs) * 2
