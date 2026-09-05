"""Kolumny pochodne liczone RAZ, centralnie, zaraz po wczytaniu z Supabase - zeby kazda
strona i kazdy wykres widzialy te same, gotowe pola (energia/probke w krotkiej nazwie,
probki/dzul, flaga whole_system, czytelny `device_short`, `arch_class`, etykieta zakresu
pomiaru). Wydzielone z queries.py, zeby dalo sie testowac offline bez klienta Supabase."""

from __future__ import annotations

import re

import pandas as pd

from dashboard.i18n.pl import (
    DEVICE_TYPE_LABELS,
    MEASUREMENT_SCOPE_LABELS,
    device_category_label,
)

_MARKS = re.compile(r"\(R\)|\(TM\)|\(r\)|\(tm\)|™|®")
_WS = re.compile(r"\s+")

# Kawalki nazw handlowych, ktore nic nie wnosza na osi wykresu.
_CPU_STRIP = ["8-Core", "Processor", "13th Gen", "12th Gen", "11th Gen", "10th Gen", "Gen"]
_CPU_VENDOR = ["AMD ", "Intel "]
_GPU_STRIP_SUFFIX = [" Laptop GPU", " Graphics"]
_GPU_VENDOR = ["NVIDIA ", "GeForce ", "AMD ", "Intel ", "Advanced Micro Devices ", "Corporation "]


def _collapse(text: str) -> str:
    return _WS.sub(" ", _MARKS.sub(" ", text or "")).strip(" -·")


def short_cpu_model(cpu_model: str | None) -> str:
    """'13th Gen Intel(R) Core(TM) i5-1335U' -> 'Core i5-1335U'."""
    if not cpu_model:
        return "CPU"
    text = _collapse(cpu_model)
    text = text.split(" with ")[0]  # 'Ryzen 7 5800HS with Radeon Graphics' -> 'Ryzen 7 5800HS'
    for token in _CPU_STRIP:
        text = re.sub(rf"\b{re.escape(token)}\b", " ", text)
    text = _collapse(text)
    for vendor in _CPU_VENDOR:
        if text.startswith(vendor):
            text = text[len(vendor):]
    text = text.replace("Core ", "Core ")  # zachowujemy 'Core'
    return _collapse(text) or "CPU"


def short_gpu_model(gpu_model: str | None, device_type: str | None) -> str:
    """'NVIDIA GeForce RTX 4060 Laptop GPU' -> 'RTX 4060';
    'Intel(R) Iris(R) Xe Graphics (iGPU)' -> 'Iris Xe iGPU'."""
    if not gpu_model:
        return DEVICE_TYPE_LABELS.get(device_type or "", device_type or "GPU")
    if device_type in ("npu_openvino", "npu_directml"):
        return _collapse(gpu_model)  # 'Intel(R) AI Boost' -> 'Intel AI Boost'

    text = _collapse(gpu_model).replace("(iGPU)", "").replace("(dGPU)", "")
    text = _collapse(text)
    for vendor in _GPU_VENDOR:
        text = text.replace(vendor, "")
    for suffix in _GPU_STRIP_SUFFIX:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = _collapse(text)
    if device_type in ("amd_igpu", "intel_gpu_openvino") and "iGPU" not in text:
        text = f"{text} iGPU"
    return text or "GPU"


def _device_short_row(row: pd.Series) -> str:
    device_type = row.get("device_device_type")
    if device_type == "cpu":
        base = short_cpu_model(row.get("device_cpu_model"))
    else:
        base = short_gpu_model(row.get("device_gpu_model"), device_type)
    return f"{base} ({device_type})"


def enrich_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Dokłada kolumny pochodne. Bezpieczne dla pustego DF i dla brakujacych kolumn
    (zdegradowany stan zamiast wyjatku - zgodnie z reszta projektu)."""
    if df.empty:
        return df

    df = df.copy()

    # --- architektura (4 klasy) - naprawiona klasyfikacja amd_igpu / npu_openvino ---
    if "device_device_type" in df.columns:
        gpu_cat = df["device_gpu_category"] if "device_gpu_category" in df.columns else None
        df["arch_class"] = [
            device_category_label(dt, gpu_cat.iloc[i] if gpu_cat is not None else None)
            for i, dt in enumerate(df["device_device_type"])
        ]
        df["device_category"] = df["arch_class"]  # alias wstecznie kompatybilny
        df["device_type_label"] = df["device_device_type"].map(
            lambda t: DEVICE_TYPE_LABELS.get(t, t)
        )

    # --- watomierz: 0.0 = brak odczytu (nie realne zero) -> NaN, spojnie wszedzie ---
    if "summary_energy_joules_smart_plug" in df.columns:
        sp = pd.to_numeric(df["summary_energy_joules_smart_plug"], errors="coerce")
        df["summary_energy_joules_smart_plug"] = sp.where(sp > 0)
    if "summary_power_discrepancy_pct" in df.columns and "summary_energy_joules_smart_plug" in df.columns:
        # rozbieznosc bez realnego odczytu watomierza jest bez znaczenia
        df.loc[df["summary_energy_joules_smart_plug"].isna(), "summary_power_discrepancy_pct"] = pd.NA

    # --- energia: krotkie nazwy + warianty pochodne ---
    if "summary_energy_per_sample_joules" in df.columns:
        eps = pd.to_numeric(df["summary_energy_per_sample_joules"], errors="coerce")
        df["energy_per_sample_j"] = eps
        df["energy_per_1000_samples_j"] = eps * 1000.0
        # probki/dzul = odwrotnosc J/probke ("im wyzej tym lepiej"); tylko dla eps > 0
        df["samples_per_joule"] = 1.0 / eps.where(eps > 0)

    # --- zakres pomiaru mocy ---
    if "measurement_scope" in df.columns:
        df["is_whole_system"] = df["measurement_scope"].eq("whole_system")
        df["scope_label"] = df["measurement_scope"].map(
            lambda s: MEASUREMENT_SCOPE_LABELS.get(s, s)
        )

    # --- czytelna, krotka etykieta urzadzenia na os X ---
    if "device_device_type" in df.columns:
        df["device_short"] = df.apply(_device_short_row, axis=1)
        # ujednoznacznienie, gdy ten sam (model, device_type) wystepuje na >1 maszynie
        if "device_machine_name" in df.columns:
            dup = df.groupby("device_short")["device_machine_name"].transform("nunique")
            mask = dup > 1
            df.loc[mask, "device_short"] = (
                df.loc[mask, "device_short"] + " · " + df.loc[mask, "device_machine_name"].astype(str)
            )

    return df
