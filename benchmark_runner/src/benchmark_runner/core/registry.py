"""Rejestr zadan (TASK_REGISTRY) - pozwala dopisac nowe zadanie bez zmian w rdzeniu runnera."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark_runner.core.base_task import BaseTask

TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Dekorator rejestrujacy klase BaseTask pod danym kluczem (uzywanym w polu 'task' configu)."""

    def decorator(cls: type["BaseTask"]) -> type["BaseTask"]:
        if name in TASK_REGISTRY:
            raise ValueError(f"Zadanie '{name}' jest juz zarejestrowane (klasa {TASK_REGISTRY[name].__name__})")
        TASK_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_task_class(name: str) -> type["BaseTask"]:
    try:
        return TASK_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY)) or "(brak zarejestrowanych zadan)"
        raise KeyError(f"Nieznane zadanie '{name}'. Dostepne: {available}") from exc
