"""Zadanie klasyfikacji sentymentu: fine-tuning + inferencja DistilBERT na IMDB.

Na razie tylko backend 'pytorch' (CPU/CUDA/ROCm) - sciezka NPU dla tego zadania
(export do ONNX, patrz onnx_export.py) jest przygotowana, ale nie wpieta jeszcze
w run_context - planowana rozbudowa poza Faza 5.
"""

from __future__ import annotations

import logging

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from benchmark_runner.core.base_task import BaseTask, TaskStepResult
from benchmark_runner.core.registry import register_task
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.metrics.flops import estimate_flops
from benchmark_runner.tasks.nlp_sentiment.data import MAX_SEQ_LENGTH, MODEL_NAME, get_imdb_dataloaders
from benchmark_runner.utils.quantization import quantize_dynamic_int8

logger = logging.getLogger(__name__)

# int8 = dynamiczna kwantyzacja PyTorch (torch.quantization.quantize_dynamic, patrz
# utils/quantization.py) - TYLKO na CPU (kernele FBGEMM/QNNPACK) i TYLKO inferencja
# (orchestrator._filter_unsupported_combos pomija int8+train przed wykonaniem). DistilBERT
# jest w wiekszosci warstwami Linear (projekcje uwagi, FFN), wiec w praktyce typowo 'success'
# (w odroznieniu od modeli konwolucyjnych w image_classification, gdzie Conv2d nie jest
# wspierany przez dynamiczna kwantyzacje PyTorch i wynik jest typowo 'partial').
_SUPPORTED_PRECISIONS = {"fp32", "fp16", "int8"}


@register_task("nlp_sentiment")
class NlpSentimentTask(BaseTask):
    def __init__(self, run_context: RunContext) -> None:
        super().__init__(run_context)
        if run_context.device.backend != "pytorch":
            raise NotImplementedError(
                f"NlpSentimentTask obsluguje na razie tylko backend 'pytorch' (CPU/CUDA/ROCm) - "
                f"backend '{run_context.device.backend}' (NPU) nie jest jeszcze wspierany dla tego zadania"
            )

        self._device = torch.device(run_context.device.torch_device_str or "cpu")
        self._use_amp = run_context.run_spec.precision == "fp16" and self._device.type == "cuda"

        self._model = None
        self._optimizer: optim.Optimizer | None = None
        self._scaler = GradScaler(device=self._device.type, enabled=self._use_amp)
        self._train_loader = None
        self._test_loader = None
        # ustawiane tylko gdy precision=='int8' (patrz prepare()) - dolaczane do
        # TaskStepResult.extra, orchestrator zapisuje w run_summary.quantization_status.
        self._quantization_status: str | None = None

        # FLOPS dla jednego przejscia forward (batch=1), liczone raz w prepare() - patrz
        # estimate_flops_per_sample() nizej i metrics/flops.py.
        self._flops_per_sample: float | None = None

    def prepare(self) -> None:
        from transformers import DistilBertForSequenceClassification

        run_spec = self.run_context.run_spec
        if run_spec.precision not in _SUPPORTED_PRECISIONS:
            logger.warning(
                "Precyzja '%s' nie jest jeszcze wspierana (wymaga kwantyzacji) - uruchamiam mimo to z fp32.",
                run_spec.precision,
            )

        # Osobna, jednorazowa instancja CPU/fp32 WYLACZNIE do oszacowania FLOPS - niezalezna
        # od faktycznego urzadzenia/precyzji przebiegu (patrz analogiczny komentarz w
        # tasks/image_classification/task.py:_prepare_pytorch). Wejscie DistilBERT to para
        # tensorow (input_ids, attention_mask), nie pojedynczy tensor float, stad dummy_inputs
        # zamiast input_shape.
        flops_model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        dummy_input_ids = torch.randint(0, flops_model.config.vocab_size, (1, MAX_SEQ_LENGTH), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, MAX_SEQ_LENGTH), dtype=torch.long)
        self._flops_per_sample = estimate_flops(flops_model, dummy_inputs=(dummy_input_ids, dummy_attention_mask))

        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(self._device)

        if run_spec.precision == "int8":
            if self._device.type != "cpu":
                # Obronny fallback - configi z int8 sa celowo tylko w CPU/ (dynamiczna
                # kwantyzacja PyTorch dziala jedynie na CPU), ale --device moze to nadpisac.
                logger.warning(
                    "Precyzja int8 wymaga backendu CPU (dynamiczna kwantyzacja PyTorch nie dziala na %s) - "
                    "uruchamiam mimo to z fp32",
                    self._device.type,
                )
                self._quantization_status = "failed"
            else:
                model.eval()
                model, self._quantization_status = quantize_dynamic_int8(model)

        self._model = model
        self._train_loader, self._test_loader = get_imdb_dataloaders(batch_size=run_spec.batch_size)
        self._optimizer = optim.AdamW(self._model.parameters(), lr=run_spec.train.learning_rate)

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
        assert self._model is not None
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)
        labels = batch["labels"].to(self._device)

        with autocast(device_type=self._device.type, enabled=self._use_amp):
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        correct = int((outputs.logits.argmax(dim=-1) == labels).sum().item())
        return outputs.loss, correct, input_ids.size(0)

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, int, int]:
        assert self._optimizer is not None
        self._optimizer.zero_grad()
        loss, correct, batch_size = self._step(batch)
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        return loss.item(), correct, batch_size

    def warmup(self, n_iters: int) -> None:
        assert self._model is not None and self._train_loader is not None
        self._model.train()
        data_iter = iter(self._train_loader)
        for _ in range(n_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self._train_loader)
                batch = next(data_iter)
            self._train_step(batch)

    def run_train_epoch(self) -> TaskStepResult:
        assert self._model is not None and self._train_loader is not None
        self._model.train()

        total_samples = 0
        total_loss = 0.0
        correct = 0

        for batch in self._train_loader:
            loss_value, batch_correct, batch_size = self._train_step(batch)
            total_samples += batch_size
            total_loss += loss_value * batch_size
            correct += batch_correct

        return TaskStepResult(
            samples_processed=total_samples,
            loss=total_loss / total_samples if total_samples else None,
            accuracy=correct / total_samples if total_samples else None,
        )

    def run_inference_pass(self) -> TaskStepResult:
        assert self._model is not None and self._test_loader is not None
        self._model.eval()

        total_samples = 0
        correct = 0
        num_iterations = self.run_context.run_spec.inference.num_iterations

        with torch.no_grad():
            for i, batch in enumerate(self._test_loader):
                if i >= num_iterations:
                    break
                _, batch_correct, batch_size = self._step(batch)
                total_samples += batch_size
                correct += batch_correct

        extra = {"quantization_status": self._quantization_status} if self._quantization_status else {}
        return TaskStepResult(
            samples_processed=total_samples,
            accuracy=correct / total_samples if total_samples else None,
            extra=extra,
        )

    def estimate_flops_per_sample(self) -> float | None:
        return self._flops_per_sample

    def teardown(self) -> None:
        self._model = None
        self._optimizer = None
        self._train_loader = None
        self._test_loader = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
