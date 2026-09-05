"""Profil urzadzenia obliczeniowego wykrytego na biezacej maszynie."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    device_type: str  # 'cpu' | 'cuda' | 'rocm' | 'npu_openvino' | 'npu_directml'
    label: str
    vendor: str | None
    torch_device_str: str | None  # np. "cuda:0"; None dla backendow spoza PyTorch (openvino/onnxruntime)
    backend: str  # 'pytorch' | 'onnxruntime' | 'openvino' | 'llama_cpp'
    driver_version: str | None = None
    # 'discrete' | 'integrated' | None - tylko dla cuda/rocm (patrz devices/gpu_classification.py),
    # None dla cpu/npu (kategoria nie ma zastosowania) - uzywane przez dashboard do rozroznienia
    # GPU dyskretnych od zintegrowanych (iGPU) w porownaniach.
    gpu_category: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
