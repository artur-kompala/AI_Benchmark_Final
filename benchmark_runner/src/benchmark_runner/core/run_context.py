"""Obiekt przekazywany przez caly przebieg (run) - konfiguracja + urzadzenie + kontekst maszyny."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from benchmark_runner.config.schema import RunConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.utils.system_info import SystemInfo


@dataclass
class RunContext:
    run_spec: RunConfig
    device: DeviceProfile
    system_info: SystemInfo
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
