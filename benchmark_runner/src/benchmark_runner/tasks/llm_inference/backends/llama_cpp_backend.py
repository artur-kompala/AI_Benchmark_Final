"""Backend inferencji LLM przez llama-cpp-python (pliki GGUF).

W przeciwienstwie do backendu ONNX Runtime, kwantyzacja jest tu "wpieczona" w sam
plik GGUF (np. F16, Q8_0 ~ int8, Q4_K_M ~ int4) - porownanie precyzji realizuje sie
wskazujac w configu (llm.gguf_model_path) rozne pliki per przebieg, a nie przez
konwersje w locie.

Uzytkownik musi samodzielnie dostarczyc plik .gguf (np. z Hugging Face Hub) - runner
go nie pobiera automatycznie, bo to duze pliki binarne wymagajace swiadomej decyzji
uzytkownika co do zrodla/rozmiaru pobrania.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LlamaCppLlmBackend:
    def __init__(self, model_path: str, n_ctx: int = 512) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._llm: Any = None

    def load(self) -> None:
        from llama_cpp import Llama

        model_path = Path(self._model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Plik modelu GGUF nie istnieje: {model_path}. Dla backendu 'llama_cpp' musisz "
                "samodzielnie pobrac plik .gguf i wskazac go w configu (llm.gguf_model_path) - "
                "patrz docs/setup dla instrukcji."
            )
        self._llm = Llama(model_path=str(model_path), n_ctx=self._n_ctx, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int) -> int:
        assert self._llm is not None
        output = self._llm(prompt, max_tokens=max_new_tokens, echo=False)
        completion_tokens = output.get("usage", {}).get("completion_tokens")
        if completion_tokens is not None:
            return int(completion_tokens)
        # fallback gdyby struktura odpowiedzi sie zmienila w przyszlej wersji llama-cpp-python
        text = output.get("choices", [{}])[0].get("text", "")
        return len(text.split())

    def close(self) -> None:
        self._llm = None
