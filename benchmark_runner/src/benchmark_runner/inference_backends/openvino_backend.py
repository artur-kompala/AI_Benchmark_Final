"""Backend inferencji przez OpenVINO - podstawowa sciezka wykonania modeli na NPU.

Modele PyTorch bez natywnego wsparcia NPU sa eksportowane do ONNX
(tasks/*/onnx_export.py) i uruchamiane tutaj przez OpenVINO z device_name="NPU".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OpenVinoInferenceBackend:
    def __init__(self, onnx_path: str | Path, device: str = "NPU") -> None:
        self._onnx_path = str(onnx_path)
        self._device = device
        self._compiled_model = None
        self._output_layer = None

    def compile(self) -> None:
        from openvino import Core

        core = Core()
        model = core.read_model(self._onnx_path)
        self._compiled_model = core.compile_model(model, device_name=self._device)
        self._output_layer = self._compiled_model.output(0)

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        if self._compiled_model is None:
            raise RuntimeError("OpenVinoInferenceBackend.compile() nie zostalo jeszcze wywolane")
        result = self._compiled_model([input_array])
        return result[self._output_layer]
