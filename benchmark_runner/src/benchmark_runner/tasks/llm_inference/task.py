"""Benchmark inferencji LLM - TYLKO inferencja (brak fazy treningu dla tego zadania,
zgodnie z domyslna implementacja BaseTask.run_train_epoch).

Dwa silniki wybierane przez `llm.engine` w configu:
- 'onnxruntime': eksport malego modelu Hugging Face do ONNX + fp16/int8 w locie.
- 'llama_cpp': lokalny plik GGUF dostarczony przez uzytkownika (kwantyzacja jest
  czescia pliku, wiec porownanie precyzji = rozne pliki per przebieg).
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_task import BaseTask, TaskStepResult
from benchmark_runner.core.registry import register_task
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.llm_inference.prompts import DEFAULT_PROMPTS

logger = logging.getLogger(__name__)


@register_task("llm_inference")
class LlmInferenceTask(BaseTask):
    def __init__(self, run_context: RunContext) -> None:
        super().__init__(run_context)
        self._engine_name = run_context.run_spec.llm.engine
        self._backend = None
        self._prompts = DEFAULT_PROMPTS

    def prepare(self) -> None:
        run_spec = self.run_context.run_spec

        if self._engine_name == "onnxruntime":
            from benchmark_runner.tasks.llm_inference.backends.onnxruntime_backend import OnnxRuntimeLlmBackend

            self._backend = OnnxRuntimeLlmBackend(
                hf_model_name=run_spec.llm.hf_model_name,
                precision=run_spec.precision,
                device=self.run_context.device,
            )
        elif self._engine_name == "llama_cpp":
            from benchmark_runner.tasks.llm_inference.backends.llama_cpp_backend import LlamaCppLlmBackend

            if not run_spec.llm.gguf_model_path:
                raise ValueError("llm.gguf_model_path musi byc ustawiony w configu dla engine='llama_cpp'")
            self._backend = LlamaCppLlmBackend(model_path=run_spec.llm.gguf_model_path)
        else:
            raise NotImplementedError(f"Nieznany silnik LLM '{self._engine_name}'")

        self._backend.load()

    def warmup(self, n_iters: int) -> None:
        assert self._backend is not None
        max_new_tokens = self.run_context.run_spec.llm.max_new_tokens
        for i in range(n_iters):
            prompt = self._prompts[i % len(self._prompts)]
            self._backend.generate(prompt, max_new_tokens=max_new_tokens)

    def run_inference_pass(self) -> TaskStepResult:
        assert self._backend is not None
        num_iterations = self.run_context.run_spec.inference.num_iterations
        max_new_tokens = self.run_context.run_spec.llm.max_new_tokens

        total_tokens = 0
        for i in range(num_iterations):
            prompt = self._prompts[i % len(self._prompts)]
            total_tokens += self._backend.generate(prompt, max_new_tokens=max_new_tokens)

        # samples_processed = liczba wygenerowanych tokenow (throughput = tokens/s) -
        # standardowa metryka dla benchmarkow inferencji LLM, w odroznieniu od
        # samples/s uzywanego dla zadan klasyfikacji.
        return TaskStepResult(samples_processed=total_tokens, extra={"unit": "tokens"})

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
