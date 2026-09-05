"""Backend inferencji przez ONNX Runtime + DirectML - alternatywna sciezka NPU, dla
operatorow/modeli niewspieranych jeszcze przez plugin NPU w OpenVINO."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxRuntimeInferenceBackend:
    def __init__(self, onnx_path: str | Path, provider: str = "DmlExecutionProvider") -> None:
        self._onnx_path = str(onnx_path)
        self._provider = provider
        self._session = None
        self._input_name = None
        self._output_name = None

    def compile(self) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(self._onnx_path, providers=[self._provider])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("OnnxRuntimeInferenceBackend.compile() nie zostalo jeszcze wywolane")
        return self._session.run([self._output_name], {self._input_name: input_array})[0]
