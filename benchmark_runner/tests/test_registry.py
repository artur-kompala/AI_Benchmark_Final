from __future__ import annotations

import pytest

from benchmark_runner.core.base_task import BaseTask, TaskStepResult
from benchmark_runner.core.registry import TASK_REGISTRY, get_task_class, register_task


class _StubTask(BaseTask):
    def prepare(self) -> None:
        pass

    def warmup(self, n_iters: int) -> None:
        pass

    def run_inference_pass(self) -> TaskStepResult:
        return TaskStepResult(samples_processed=1)


def test_register_and_get_task_class():
    @register_task("test_dummy_task_registry")
    class DummyTask(_StubTask):
        pass

    try:
        assert get_task_class("test_dummy_task_registry") is DummyTask
        assert DummyTask.name == "test_dummy_task_registry"
    finally:
        del TASK_REGISTRY["test_dummy_task_registry"]


def test_get_unknown_task_raises_key_error():
    with pytest.raises(KeyError):
        get_task_class("does_not_exist_task_xyz")


def test_register_duplicate_task_name_raises():
    @register_task("test_dummy_dup")
    class DummyA(_StubTask):
        pass

    try:
        with pytest.raises(ValueError):

            @register_task("test_dummy_dup")
            class DummyB(_StubTask):
                pass

    finally:
        del TASK_REGISTRY["test_dummy_dup"]
