"""Testy kolumn pochodnych (data_access/enrich.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.data_access.enrich import enrich_runs_df, short_cpu_model, short_gpu_model


def _raw_df():
    return pd.DataFrame(
        {
            "device_device_type": ["cpu", "cuda", "amd_igpu", "npu_openvino"],
            "device_gpu_category": [None, "discrete", "integrated", None],
            "device_machine_name": ["m1", "m1", "m2", "m3"],
            "device_cpu_model": ["13th Gen Intel(R) Core(TM) i5-1335U", "AMD Ryzen 7 5700X3D 8-Core Processor",
                                 "AMD Ryzen 7 5800HS with Radeon Graphics", "Intel(R) Core(TM) Ultra 7 155H"],
            "device_gpu_model": [None, "NVIDIA GeForce RTX 4060 Laptop GPU", "AMD Radeon 780M Graphics",
                                 "Intel(R) AI Boost"],
            "measurement_scope": ["device_only", "device_only", "whole_system", "whole_system"],
            "summary_energy_per_sample_joules": [0.4, 0.02, 0.25, 0.03],
            "summary_energy_joules_smart_plug": [90.0, 0.0, None, None],
            "summary_power_discrepancy_pct": [52.0, 41.0, None, None],
        }
    )


def test_short_cpu_model_strips_marketing_noise():
    assert short_cpu_model("13th Gen Intel(R) Core(TM) i5-1335U") == "Core i5-1335U"
    assert short_cpu_model("AMD Ryzen 7 5700X3D 8-Core Processor") == "Ryzen 7 5700X3D"
    assert short_cpu_model("AMD Ryzen 7 5800HS with Radeon Graphics") == "Ryzen 7 5800HS"
    assert short_cpu_model(None) == "CPU"


def test_short_gpu_model_strips_marketing_noise():
    assert short_gpu_model("NVIDIA GeForce RTX 4060 Laptop GPU", "cuda") == "RTX 4060"
    assert short_gpu_model("Intel(R) Iris(R) Xe Graphics (iGPU)", "intel_gpu_openvino") == "Iris Xe iGPU"
    assert short_gpu_model("AMD Radeon 780M Graphics", "amd_igpu") == "Radeon 780M iGPU"
    assert short_gpu_model("Intel(R) AI Boost", "npu_openvino") == "Intel AI Boost"


def test_enrich_adds_derived_columns():
    out = enrich_runs_df(_raw_df())
    assert list(out["arch_class"]) == [
        "CPU", "GPU dyskretne", "GPU zintegrowane (iGPU)", "NPU (akcelerator dedykowany)",
    ]
    assert list(out["is_whole_system"]) == [False, False, True, True]
    # samples_per_joule = 1 / energia-na-probke
    assert np.isclose(out.loc[0, "samples_per_joule"], 1 / 0.4)
    assert np.isclose(out.loc[1, "energy_per_1000_samples_j"], 20.0)
    assert "(cpu)" in out.loc[0, "device_short"]


def test_enrich_smart_plug_zero_becomes_nan_and_clears_discrepancy():
    out = enrich_runs_df(_raw_df())
    # wiersz cuda ma smart_plug = 0.0 -> NaN, a wiec i discrepancy wyzerowane
    assert pd.isna(out.loc[1, "summary_energy_joules_smart_plug"])
    assert pd.isna(out.loc[1, "summary_power_discrepancy_pct"])
    # wiersz cpu ma realny odczyt -> zostaje
    assert out.loc[0, "summary_energy_joules_smart_plug"] == 90.0
    assert out.loc[0, "summary_power_discrepancy_pct"] == 52.0


def test_enrich_empty_df_is_safe():
    assert enrich_runs_df(pd.DataFrame()).empty


def test_enrich_disambiguates_duplicate_device_short_by_machine():
    raw = pd.DataFrame({
        "device_device_type": ["intel_gpu_openvino", "intel_gpu_openvino"],
        "device_gpu_category": ["integrated", "integrated"],
        "device_machine_name": ["business-dell", "kamila"],
        "device_cpu_model": ["x", "y"],
        "device_gpu_model": ["Intel(R) Iris(R) Xe Graphics (iGPU)", "Intel(R) Iris(R) Xe Graphics (iGPU)"],
        "measurement_scope": ["whole_system", "whole_system"],
        "summary_energy_per_sample_joules": [0.03, 0.03],
    })
    out = enrich_runs_df(raw)
    assert out.loc[0, "device_short"] != out.loc[1, "device_short"]
    assert "business-dell" in out.loc[0, "device_short"]
