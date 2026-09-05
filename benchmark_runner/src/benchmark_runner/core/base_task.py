"""Abstrakcyjna klasa bazowa dla zadan benchmarkowych (image_classification, nlp_sentiment, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark_runner.core.run_context import RunContext


@dataclass
class TaskStepResult:
    samples_processed: int
    loss: float | None = None
    accuracy: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Kazde zadanie (task) implementuje ten interfejs i rejestruje sie
    przez dekorator @register_task w core/registry.py."""

    name: str

    def __init__(self, run_context: "RunContext") -> None:
        self.run_context = run_context

    @abstractmethod
    def prepare(self) -> None:
        """Wczytanie danych, budowa modelu, ewentualny eksport do ONNX dla NPU."""

    @abstractmethod
    def warmup(self, n_iters: int) -> None:
        """Kilka iteracji bez pomiaru - wyklucza cold-start/throttling ze wlasciwego pomiaru."""

    def run_train_epoch(self) -> TaskStepResult:
        """Domyslnie brak wsparcia treningu; zadania trenowalne nadpisuja te metode."""
        raise NotImplementedError(f"{self.__class__.__name__} nie wspiera fazy 'train'")

    @abstractmethod
    def run_inference_pass(self) -> TaskStepResult:
        """Jedno przejscie inferencji (np. przez zbior testowy lub N iteracji)."""

    def estimate_flops_per_sample(self) -> float | None:
        """FLOPS dla jednego przejscia forward (batch=1), na potrzeby run_summary.
        flops_per_sample/flops_per_watt - patrz metrics/flops.py. Domyslnie brak wsparcia
        (None, zadanie nie przerywa benchmarku); zadania z modelem PyTorch o prostym
        ksztalcie wejscia nadpisuja te metode (patrz ImageClassificationTask,
        NlpSentimentTask)."""
        return None

    def teardown(self) -> None:
        """Zwolnienie zasobow. Domyslnie no-op."""
        return None
