"""Wspolne atrapy i wytwornice danych do testow offline (bez sieci, bez Supabase)."""

from __future__ import annotations

import pandas as pd
import pytest


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    """Atrapa builder-a supabase-py: obsluguje .select/.eq/.order/.range/.execute i
    stronicowanie przez .range(start, end) (inclusive) - jak PostgREST."""

    def __init__(self, rows):
        self._rows = list(rows)
        self._start = 0
        self._end = None

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def range(self, start, end):
        self._start, self._end = start, end
        return self

    def execute(self):
        if self._end is None:
            return FakeResponse(list(self._rows))
        return FakeResponse(self._rows[self._start : self._end + 1])


class FakeClient:
    def __init__(self, rows_by_table):
        self._rows_by_table = rows_by_table

    def table(self, name):
        return FakeQuery(self._rows_by_table.get(name, []))


def make_runs_rows(n_devices: int = 2, reps: int = 3, extra: list[dict] | None = None) -> list[dict]:
    """Buduje surowe wiersze `runs` w ksztalcie odpowiedzi PostgREST (z zagniezdzonymi
    devices(*) i run_summary(*)). Domyslnie: 2 urzadzenia (cpu + cuda), po `reps` powtorzen
    jednej kombinacji."""
    devices = [
        {"id": "dev-cpu", "machine_name": "m1", "device_type": "cpu", "device_label": "CPU",
         "vendor": None, "gpu_category": None, "cpu_model": "Test Ryzen 7 1234X", "gpu_model": None},
        {"id": "dev-cuda", "machine_name": "m1", "device_type": "cuda", "device_label": "RTX Test",
         "vendor": "nvidia", "gpu_category": "discrete", "cpu_model": "Test Ryzen 7 1234X",
         "gpu_model": "NVIDIA GeForce RTX 4060 Laptop GPU"},
        {"id": "dev-amdigpu", "machine_name": "m2", "device_type": "amd_igpu", "device_label": "Radeon iGPU",
         "vendor": "amd", "gpu_category": "integrated", "cpu_model": "Test Ryzen 5 5600HS",
         "gpu_model": "AMD Radeon Graphics"},
        {"id": "dev-npu", "machine_name": "m3", "device_type": "npu_openvino", "device_label": "Intel AI Boost",
         "vendor": "intel", "gpu_category": None, "cpu_model": "Test Core Ultra 7 155H",
         "gpu_model": "Intel(R) AI Boost"},
    ][:n_devices]

    rows = []
    rid = 0
    for di, dev in enumerate(devices):
        scope = "device_only" if dev["device_type"] in ("cpu", "cuda", "rocm") else "whole_system"
        for r in range(reps):
            rid += 1
            eps = 0.10 * (di + 1) + 0.01 * r  # rosnie z indeksem urzadzenia
            rows.append({
                "id": f"run-{rid}",
                "device_id": dev["id"],
                "task": "image_classification",
                "model_name": "mobilenet_v3",
                "phase": "inference",
                "batch_size": 4,
                "precision": "fp32",
                "measurement_scope": scope,
                "status": "completed",
                "started_at": f"2026-09-0{(rid % 9) + 1}T10:00:00+00:00",
                "duration_s": 1.0 + r,
                "throughput_samples_per_s": 100.0 * (di + 1),
                "avg_temperature_c": 55.0 + di,
                "repetition_group_id": f"grp-{dev['id']}",
                "repetition_index": r + 1,
                "devices": dict(dev),
                "run_summary": {
                    "run_id": f"run-{rid}",
                    "energy_joules_software": 40.0 * (di + 1) + r,
                    "energy_joules_smart_plug": (0.0 if di == 1 else 90.0 * (di + 1) + r),
                    "power_discrepancy_pct": (None if di != 0 else 50.0 + r),
                    "energy_per_sample_joules": eps,
                    "avg_power_watts": 30.0 + 10 * di,
                    "peak_power_watts": 45.0 + 12 * di,
                    "flops_per_watt": 1.0e7 * (di + 1),
                    "accuracy": 0.44,
                    "samples_processed": 100,
                    "power_samples_count": 25 if r else 10,
                    "quantization_status": None,
                },
            })
    for e in extra or []:
        rows.append(e)
    return rows


@pytest.fixture
def runs_rows():
    return make_runs_rows(n_devices=4, reps=3)
