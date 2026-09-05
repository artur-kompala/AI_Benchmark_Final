"""Testy analizy (data_access/aggregation.py): agregacja dwustopniowa, front Pareto,
tabela decyzyjna, porownania w obrebie maszyny, przeciecia krzywych, listy dynamiczne."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.data_access.aggregation import (
    SMALL_N_THRESHOLD,
    agg_metric,
    aggregate_by_repetition_group,
    build_decision_table,
    coverage_matrix,
    crossover_batches,
    list_device_types,
    list_machines,
    pareto_front,
    same_machine_winners,
)
from dashboard.data_access.enrich import enrich_runs_df
from dashboard.data_access.queries import get_runs_df
from tests.conftest import FakeClient, make_runs_rows


def _df():
    return get_runs_df(FakeClient({"runs": make_runs_rows(n_devices=4, reps=3)}))


# --- agregacja po grupie powtorzen -----------------------------------------
def test_aggregate_by_repetition_group_counts_reps_and_fills_std():
    df = _df()
    agg = aggregate_by_repetition_group(df, "energy_per_sample_j", "m", "s")
    assert set(agg["n_repetitions"]) == {3}
    assert (agg["s"] >= 0).all()  # std policzone, nie NaN


def test_agg_metric_two_stage_and_small_n_flag():
    df = _df()
    out = agg_metric(df, "energy_per_sample_j", by=["device_device_type"])
    # 1 grupa powtorzen na device_type -> n == 1 -> small_n True
    assert (out["n"] == 1).all()
    assert out["small_n"].all()
    assert (out["n_runs"] == 3).all()
    assert SMALL_N_THRESHOLD == 3


def test_agg_metric_respects_by_columns_and_ignores_missing():
    df = _df()
    out = agg_metric(df, "energy_per_sample_j", by=["device_device_type", "nie_ma_takiej"])
    assert "device_device_type" in out.columns and len(out) == 4


# --- front Pareto ---------------------------------------------------------
def test_pareto_front_marks_non_dominated_only():
    d = pd.DataFrame({
        "thr": [10, 20, 30, 15],
        "eps": [1.0, 0.5, 0.2, 2.0],   # wiersz 3 dominuje wszystko; wiersz 0 tez na froncie (najmniej thr)
    })
    front = pareto_front(d, "thr", "eps", x_better="higher", y_better="lower")
    assert bool(front.iloc[2]) is True         # najszybszy i najoszczedniejszy
    assert bool(front.iloc[3]) is False        # zdominowany przez kazdy inny


def test_pareto_front_empty_input():
    assert pareto_front(pd.DataFrame({"a": [], "b": []}), "a", "b").empty


# --- tabela decyzyjna ---------------------------------------------------
def test_build_decision_table_picks_lowest_energy_winner():
    df = _df()  # eps rosnie z indeksem urzadzenia -> cpu (dev 0) najnizej
    dt = build_decision_table(df, group_col="device_type_label")
    assert len(dt) == 1
    row = dt.iloc[0]
    assert row["zwyciezca"] == "CPU"
    assert row["przewaga_pct"] > 0
    assert row["ma_whole_system"] is True or bool(row["ma_whole_system"]) is True  # amd_igpu/npu w grze


def test_build_decision_table_device_types_filter():
    df = _df()
    dt = build_decision_table(df, device_types=["cuda", "amd_igpu"])
    assert dt.iloc[0]["zwyciezca"] == "GPU NVIDIA (CUDA)"  # nizsze eps niz amd_igpu


def test_build_decision_table_excludes_llm_inference():
    rows = make_runs_rows(n_devices=2, reps=2)
    for r in rows:
        r["task"] = "llm_inference"
    df = get_runs_df(FakeClient({"runs": rows}))
    assert build_decision_table(df).empty


# --- porownania w obrebie maszyny -------------------------------------
def test_same_machine_winners_only_keeps_multi_device_scenarios():
    df = _df()  # m1 ma cpu+cuda (2 urzadzenia); m2, m3 po jednym
    w = same_machine_winners(df)
    assert set(w["device_machine_name"]) == {"m1"}
    assert w.iloc[0]["zwyciezca"] == "CPU"
    assert bool(w.iloc[0]["cpu_wygral"]) is True


# --- przeciecia krzywych ------------------------------------------------
def test_crossover_batches_detects_sign_flip():
    line = pd.DataFrame({
        "device_device_type": ["cpu", "cpu", "rocm", "rocm"],
        "x": [1, 4, 1, 4],
        "y": [0.5, 0.5, 0.4, 0.6],  # cpu-rocm: -0.1 -> +(-0.1)? cpu<rocm@4, cpu>rocm@1
    })
    cr = crossover_batches(line, "x", "y", "device_device_type", reference="cpu")
    assert len(cr) == 1 and cr[0]["seria"] == "rocm"
    assert 1 <= cr[0]["x_przeciecia"] <= 4


# --- listy dynamiczne / heatmapa pokrycia ----------------------------
def test_list_helpers_are_derived_from_dataframe():
    df = _df()
    assert list_machines(df) == ["m1", "m2", "m3"]
    dts = list_device_types(df, order=["cpu", "cuda", "rocm", "amd_igpu", "intel_gpu_openvino", "npu_openvino"])
    assert dts[:2] == ["cpu", "cuda"] and "npu_openvino" in dts


def test_coverage_matrix_counts_runs():
    df = _df()
    cm = coverage_matrix(df, index="device_type_label", columns=["task", "phase"])
    assert cm.to_numpy().sum() == len(df)
    assert (cm.to_numpy() >= 0).all()


def test_new_machine_appears_without_code_change():
    """Symuluje dojscie maszyny 'wojtek' - listy musza sie zaktualizowac same."""
    rows = make_runs_rows(n_devices=2, reps=2)
    extra = make_runs_rows(n_devices=2, reps=2)
    for r in extra:
        r["id"] += "-w"
        r["devices"] = dict(r["devices"], id=r["devices"]["id"] + "-w", machine_name="wojtek")
        r["device_id"] += "-w"
        r["repetition_group_id"] += "-w"
    df = get_runs_df(FakeClient({"runs": rows + extra}))
    assert "wojtek" in list_machines(df)
