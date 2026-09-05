"""Walidacja configu dla LlmInferenceTask.prepare() - bez pobierania modeli (szybkie testy)."""

from __future__ import annotations

import pytest

from benchmark_runner.config.schema import ExperimentConfig, LlmConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.llm_inference.task import LlmInferenceTask
from benchmark_runner.utils.system_info import SystemInfo


def _make_run_context(llm_config: LlmConfig) -> RunContext:
    experiment = ExperimentConfig(
        experiment_name="test",
        task="llm_inference",
        models=["tiny-gpt2"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[1],
        phases=["inference"],
        llm=llm_config,
    )
    run_spec = experiment.expand_matrix()[0]

    device = DeviceProfile(device_type="cpu", label="CPU", vendor=None, torch_device_str="cpu", backend="pytorch")
    system_info = SystemInfo(
        machine_name="test-machine",
        os_name="Windows",
        os_version="10",
        cpu_model="Test CPU",
        total_ram_gb=16.0,
    )
    return RunContext(run_spec=run_spec, device=device, system_info=system_info)


def test_prepare_raises_when_llama_cpp_engine_missing_model_path():
    run_context = _make_run_context(LlmConfig(engine="llama_cpp", gguf_model_path=None))
    task = LlmInferenceTask(run_context)

    with pytest.raises(ValueError, match="gguf_model_path"):
        task.prepare()


def test_llm_config_rejects_unknown_engine():
    with pytest.raises(Exception):
        LlmConfig(engine="not_a_real_engine")


def test_default_engine_is_onnxruntime():
    run_context = _make_run_context(LlmConfig())
    assert run_context.run_spec.llm.engine == "onnxruntime"
