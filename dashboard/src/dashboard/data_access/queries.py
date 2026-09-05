"""Zapytania read-only do Supabase. Dashboard NIGDY nie zapisuje danych - tylko czyta
wyniki wygenerowane przez benchmark_runner (osobny, niezalezny proces)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from supabase import Client

from dashboard.data_access.enrich import enrich_runs_df
from dashboard.data_access.supabase_client import get_supabase_client
from dashboard.i18n.pl import DEVICE_TYPE_LABELS, device_category_label

_TIMESTAMP_COLUMNS = ("started_at", "finished_at", "created_at", "computed_at")


def _flatten_run_row(row: dict[str, Any]) -> dict[str, Any]:
    """Splaszcza zagniezdzone obiekty devices(*)/run_summary(*) zwrocone przez
    PostgREST resource embedding do plaskich kolumn device_*/summary_*.

    run_summary.run_id jest kluczem glownym (relacja 1:1) - PostgREST embeduje taka
    relacje jako pojedynczy obiekt (dict) albo None, NIE jako liste (w odroznieniu od
    devices, ktore jest relacja N:1 zwracana zawsze jako dict tez, ale technicznie
    mogloby byc lista przy innej kardynalnosci - stad ta sama defensywna obsluga obu)."""
    device = row.pop("devices", None) or {}
    if isinstance(device, list):
        device = device[0] if device else {}

    summary = row.pop("run_summary", None) or {}
    if isinstance(summary, list):
        summary = summary[0] if summary else {}

    flat = dict(row)
    for key, value in device.items():
        if key == "id":
            flat["device_row_id"] = value
        else:
            flat[f"device_{key}"] = value
    for key, value in summary.items():
        if key == "run_id":
            continue
        flat[f"summary_{key}"] = value
    return flat


_PAGE_SIZE = 1000  # PostgREST zwraca maks. 1000 wierszy na zapytanie - stad stronicowanie


def _fetch_all(query_factory, page_size: int = _PAGE_SIZE) -> list[dict[str, Any]]:
    """Sciaga WSZYSTKIE wiersze, obchodzac limit 1000 PostgREST przez .range().
    `query_factory` buduje swiezy obiekt zapytania (bez .execute()) przy kazdym wywolaniu -
    dane wciaz przyrastaja, wiec dashboard musi widziec pelny zbior, nie pierwszy tysiac."""
    all_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = query_factory().range(offset, offset + page_size - 1).execute()
        batch = response.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            return all_rows
        offset += page_size


def get_runs_df(client: Client) -> pd.DataFrame:
    """Wszystkie przebiegi z dolaczonymi danymi urzadzenia (devices) i metrykami
    pochodnymi (run_summary) - jeden wiersz na przebieg, gotowy do analizy w pandas."""
    rows = _fetch_all(
        lambda: client.table("runs")
        .select("*, devices(*), run_summary(*)")
        .order("started_at", desc=True)
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([_flatten_run_row(dict(row)) for row in rows])
    for col in _TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return enrich_runs_df(df)


def get_power_samples_df(client: Client, run_id: str) -> pd.DataFrame:
    """Surowe probki mocy dla jednego przebiegu, posortowane chronologicznie."""
    response = client.table("power_samples").select("*").eq("run_id", run_id).order("timestamp").execute()
    df = pd.DataFrame(response.data or [])
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def get_devices_df(client: Client) -> pd.DataFrame:
    response = client.table("devices").select("*").execute()
    df = pd.DataFrame(response.data or [])
    if not df.empty and "device_type" in df.columns:
        gpu_category_col = df["gpu_category"] if "gpu_category" in df.columns else None
        df["device_category"] = [
            device_category_label(device_type, gpu_category_col.iloc[i] if gpu_category_col is not None else None)
            for i, device_type in enumerate(df["device_type"])
        ]
    return df


def get_run_stats_summary_df(client: Client) -> pd.DataFrame:
    """run_stats_summary (Poprawka 2 - mean/std/CV liczone przez runner po kazdej grupie
    powtorzen) z dolaczona etykieta urzadzenia, do wglaadu/eksportu obok statystyk
    liczonych na biezaco w dashboardzie z surowych runs (patrz data_access/aggregation.py)."""
    rows = _fetch_all(
        lambda: client.table("run_stats_summary").select("*, devices(*)").order("computed_at", desc=True)
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([_flatten_run_row(dict(row)) for row in rows])
    if "computed_at" in df.columns:
        df["computed_at"] = pd.to_datetime(df["computed_at"], errors="coerce", utc=True)
    if "device_device_type" in df.columns:
        gpu_category_col = df["device_gpu_category"] if "device_gpu_category" in df.columns else None
        df["device_category"] = [
            device_category_label(
                device_type,
                gpu_category_col.iloc[i] if gpu_category_col is not None else None,
            )
            for i, device_type in enumerate(df["device_device_type"])
        ]
        df["arch_class"] = df["device_category"]
        df["device_type_label"] = df["device_device_type"].map(DEVICE_TYPE_LABELS)
    return df


@st.cache_data(ttl=60, show_spinner="Wczytywanie wynikow z Supabase...")
def load_runs_df_cached() -> pd.DataFrame:
    """Wspolny, cache'owany punkt wejscia uzywany przez app.py i wszystkie strony -
    unika powtarzania tego samego zapytania przy kazdym przelaczeniu strony."""
    client = get_supabase_client()
    return get_runs_df(client)


@st.cache_data(ttl=60, show_spinner="Wczytywanie statystyk powtorzen z Supabase...")
def load_run_stats_summary_df_cached() -> pd.DataFrame:
    client = get_supabase_client()
    return get_run_stats_summary_df(client)


@st.cache_data(ttl=60, show_spinner="Wczytywanie urzadzen z Supabase...")
def load_devices_df_cached() -> pd.DataFrame:
    client = get_supabase_client()
    return get_devices_df(client)
