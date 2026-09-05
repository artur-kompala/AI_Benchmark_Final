"""Eksport DistilBERT do ONNX - przygotowanie pod przyszla sciezke NPU dla tego zadania.

Nie wpiete jeszcze w task.py: NlpSentimentTask obsluguje na razie tylko backend
'pytorch' (patrz docstring w task.py) - to jest grunt pod kolejna iteracje,
analogicznie do tasks/image_classification/onnx_export.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    batch_size: int,
    seq_length: int,
    output_path: str | Path,
    opset_version: int = 18,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=opset_version,
            dynamo=True,
            verbose=False,
        )
    finally:
        model.train(was_training)

    logger.info("Wyeksportowano model do ONNX: %s", output_path)
    return output_path
