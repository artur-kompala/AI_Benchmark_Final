"""Backend inferencji LLM przez ONNX Runtime.

Eksportuje maly model przyczynowy (causal LM) z Hugging Face do ONNX i opcjonalnie
przeksztalca wg precyzji:
- fp32: model bazowy, bez zmian.
- fp16: konwersja wag do float16 (onnxconverter-common). Realne przyspieszenie fp16
  wymaga GPU/NPU execution providera - na czystym CPU dziala jako emulacja (moze byc
  WOLNIEJSZA niz fp32), ale pozwala porownac rozmiar modelu/energie miedzy precyzjami.
- int8: dynamiczna kwantyzacja wag (onnxruntime.quantization.quantize_dynamic) -
  nie wymaga danych kalibracyjnych, dziala dobrze na CPU.
- int4: NIEWSPIERANE jeszcze (wymaga wyspecjalizowanych narzedzi typu
  MatMulNBitsQuantizer) - log ostrzezenia, uruchomienie z fp32.

Dekodowanie jest naiwne (greedy, bez KV-cache) - wystarczajace do pomiaru czasu/mocy
dla malych modeli testowych, nie jest to silnik produkcyjny.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_PRECISIONS = {"fp32", "fp16", "int8"}


class OnnxRuntimeLlmBackend:
    def __init__(self, hf_model_name: str, precision: str, device: Any) -> None:
        self._hf_model_name = hf_model_name
        self._precision = precision
        self._device = device
        self._session = None
        self._tokenizer = None
        self._exported_paths: list[Path] = []

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._precision not in _SUPPORTED_PRECISIONS:
            logger.warning(
                "Precyzja '%s' nie jest jeszcze wspierana przez backend ONNX Runtime LLM "
                "(int4 wymaga MatMulNBitsQuantizer, planowane w kolejnej iteracji) - uruchamiam z fp32.",
                self._precision,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self._hf_model_name)
        model.eval()
        # use_cache=False: nasza petla generate() jest naiwna (bez KV-cache, przetwarza
        # cala rosnaca sekwencje przy kazdym kroku), a domyslnie transformers zwraca w
        # wyjsciu obiekt DynamicCache, ktorego nowy eksporter torch.onnx (torch.export)
        # nie potrafi obsluzyc ("not a known pytree type"). Wylaczenie use_cache usuwa
        # ten obiekt z wyjscia i jest zgodne z tym, jak i tak uzywamy modelu.
        model.config.use_cache = False

        base_onnx_path = Path(tempfile.gettempdir()) / f"benchmark_runner_llm_{self._hf_model_name.replace('/', '_')}.onnx"
        self._export_to_onnx(model, base_onnx_path)
        self._exported_paths.append(base_onnx_path)

        onnx_path = base_onnx_path
        if self._precision == "int8":
            onnx_path = self._quantize_int8(base_onnx_path)
            self._exported_paths.append(onnx_path)
        elif self._precision == "fp16":
            onnx_path = self._convert_fp16(base_onnx_path)
            self._exported_paths.append(onnx_path)

        import onnxruntime as ort

        self._session = ort.InferenceSession(str(onnx_path), providers=[self._resolve_provider()])

    def _resolve_provider(self) -> str:
        backend_name = getattr(self._device, "backend", "pytorch")
        if backend_name == "onnxruntime":
            return "DmlExecutionProvider"
        return "CPUExecutionProvider"

    @staticmethod
    def _export_to_onnx(model: Any, output_path: Path) -> None:
        import torch

        dummy_input = torch.randint(0, model.config.vocab_size, (1, 8), dtype=torch.long)
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
            opset_version=18,
            dynamo=True,
            verbose=False,
        )
        logger.info("Wyeksportowano model LLM do ONNX: %s", output_path)

    @staticmethod
    def _quantize_int8(onnx_path: Path) -> Path:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
        logger.info("Skwantyzowano model LLM do int8: %s", quantized_path)
        return quantized_path

    @staticmethod
    def _convert_fp16(onnx_path: Path) -> Path:
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(str(onnx_path))
        model_fp16 = float16.convert_float_to_float16(model)
        fp16_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
        onnx.save(model_fp16, str(fp16_path))
        logger.info("Skonwertowano model LLM do fp16: %s", fp16_path)
        return fp16_path

    def generate(self, prompt: str, max_new_tokens: int) -> int:
        assert self._session is not None and self._tokenizer is not None
        input_ids = self._tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int64)

        eos_token_id = self._tokenizer.eos_token_id
        generated = 0
        for _ in range(max_new_tokens):
            outputs = self._session.run(None, {"input_ids": input_ids})
            next_token_logits = outputs[0][:, -1, :]
            next_token = int(np.argmax(next_token_logits, axis=-1)[0])
            input_ids = np.concatenate([input_ids, np.array([[next_token]], dtype=np.int64)], axis=1)
            generated += 1
            if eos_token_id is not None and next_token == eos_token_id:
                break
        return generated

    def close(self) -> None:
        self._session = None
        self._tokenizer = None
        for path in self._exported_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Nie udalo sie usunac tymczasowego pliku ONNX %s", path, exc_info=True)
        self._exported_paths = []
