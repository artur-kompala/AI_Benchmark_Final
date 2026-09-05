"""Strona 2 - Wydajnosc (surowa szybkosc): przepustowosc [probki/s] i czas [s] wg
architektury. Kontrast wzgledem strony 1 - tu GPU ma zwykle wyrazna przewage."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import bar_metric_by_device
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.data_access.aggregation import agg_metric, list_tasks
from dashboard.i18n.pl import TASK_LABELS, label, phase_label

INTRO = """
**Ile probek na sekunde** przetwarza kazda architektura i **ile trwa** cale zadanie.
To jest strona o surowej szybkosci - swiadomie zestawiona z efektywnoscia energetyczna
ze strony 1, zeby pokazac, ze *szybciej* nie znaczy automatycznie *oszczedniej*.
"""

header("Wydajnosc (przepustowosc i czas)", INTRO)
df = load_runs_or_stop()
df = df[df["task"] != "llm_inference"]

tasks = list_tasks(df, order=list(TASK_LABELS))
c1, c2 = st.columns(2)
with c1:
    precisions = sorted(df["precision"].dropna().unique())
    precision = st.radio("Precyzja", precisions, index=precisions.index("fp32") if "fp32" in precisions else 0, horizontal=True)
with c2:
    phases = sorted(df["phase"].dropna().unique())
    phase = st.radio("Faza", phases, index=phases.index("inference") if "inference" in phases else 0,
                     horizontal=True, format_func=phase_label)

render_device_legend(sorted(df["device_device_type"].dropna().unique()))

scope = df[(df["precision"] == precision) & (df["phase"] == phase)]
if scope.empty:
    st.warning(f"Brak przebiegow dla {precision} / {phase_label(phase)}.")
    st.stop()

by = ["device_device_type", "device_type_label", "arch_class", "task"]

fig_thr = bar_metric_by_device(
    agg_metric(scope, "throughput_samples_per_s", by=by, per_group_first=False),
    value_col="mean", error_col="std", facet_col="task", category_order=tasks,
    y_title=label("throughput_samples_per_s"), log_y=True,
    title=f"Przepustowosc wg architektury — {precision}, {phase_label(phase)}",
)
figure_block(
    fig_thr,
    name_parts=["wydajnosc_przepustowosc", precision, phase],
    key="thr",
    co_pokazuje=(
        f"Srednia **przepustowosc** (probki/s) dla `{precision}`, faza *{phase_label(phase)}*, panel na "
        "zadanie. Skala X logarytmiczna. Wasy = odchylenie std. miedzy konfiguracjami."
    ),
    jak_czytac="Dluzszy slupek = szybciej. Serie szrafowane (iGPU/NPU) mierzone jako `whole_system` - "
    "przepustowosc jest porownywalna wprost (to nie pomiar mocy), ale te urzadzenia robia tylko `inference`.",
    co_wynika=(
        "GPU dyskretne (CUDA) i NPU zwykle wyrazenie wygrywaja surowa szybkosc - czesto o rzad "
        "wielkosci nad CPU. To jest ta \"nizsza maksymalna wydajnosc CPU\", o ktorej mowi hipoteza."
    ),
    hipoteza=(
        "Ten wykres pokazuje strone, ktora hipoteza **przyznaje** GPU: wyzsza maksymalna wydajnosc. "
        "Pytanie pracy brzmi, czy ta przewaga przeklada sie na energie - odpowiedz na stronach 1, 3, 4."
    ),
)

fig_dur = bar_metric_by_device(
    agg_metric(scope, "duration_s", by=by, per_group_first=False),
    value_col="mean", error_col="std", facet_col="task", category_order=tasks,
    y_title=label("duration_s"), log_y=True,
    title=f"Czas wykonania zadania — {precision}, {phase_label(phase)}",
)
figure_block(
    fig_dur,
    name_parts=["wydajnosc_czas", precision, phase],
    key="dur",
    co_pokazuje="Sredni **czas** pojedynczego przebiegu (s). Uwaga: liczba iteracji/probek bywa rozna "
    "miedzy konfiguracjami, wiec to zestawienie pogladowe - metryka rozstrzygajaca to przepustowosc wyzej.",
    jak_czytac="Krotszy slupek = szybciej gotowe. Skala logarytmiczna.",
    co_wynika="Czas idzie w parze z przepustowoscia - GPU konczy szybciej. Kluczowe pytanie: jaka moc "
    "chwilowa utrzymywalo urzadzenie przez ten czas (strona 6) i ile z tego wyszlo energii (strona 1).",
)

st.subheader("Dane zrodlowe (agregaty)")
tbl = agg_metric(scope, "throughput_samples_per_s", by=by, per_group_first=False)
table_block(
    tbl[[c for c in ["task", "device_type_label", "arch_class", "mean", "std", "n", "small_n"] if c in tbl.columns]],
    name_parts=["wydajnosc_tabela", precision, phase], key="thr_tbl",
    caption="mean/std przepustowosci po wszystkich przebiegach danej architektury w tym zadaniu/precyzji.",
)
growing_note()
