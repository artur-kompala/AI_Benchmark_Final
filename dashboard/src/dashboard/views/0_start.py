"""Strona 0 - Start / Jak czytac ten dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import coverage_heatmap
from dashboard.components.export_widgets import render_figure_export_buttons
from dashboard.components.narrative import interpretation
from dashboard.components.page import (
    growing_note,
    header,
    load_runs_or_stop,
    render_device_legend,
    table_block,
)
from dashboard.data_access.aggregation import coverage_matrix
from dashboard.i18n.pl import SCOPE_EXPLAINER

INTRO = """
Ten dashboard porownuje **efektywnosc energetyczna i wydajnosc** roznych architektur
obliczeniowych (CPU, GPU dyskretne, GPU zintegrowane/iGPU, NPU) w zadaniach uczenia
maszynowego **w warunkach ograniczonych zasobow** - na laptopach i domowych PC, nie w
centrach danych.

**Hipoteza pracy do zweryfikowania:** efektywnosc energetyczna istotnie zalezy od
architektury i parametrow wykonania; **dla modeli o umiarkowanej zlozonosci i malych
rozmiarach partii (batch size) CPU moze miec porownywalna lub wyzsza efektywnosc
energetyczna niz GPU**, mimo nizszej maksymalnej wydajnosci. Metryki rozstrzygajace:
**energia na probke [J/probke]** oraz energia na cale zadanie; sedno to zaleznosc
*czas -> moc chwilowa -> energia sumaryczna*.

Kazda zakladka (menu po lewej) to jeden watek analizy. Pod **kazdym** wykresem jest blok
**Co pokazuje / Jak czytac / Co z tego wynika** z odniesieniem do hipotezy.
"""

header("Analiza energooszczednosci: CPU vs GPU vs NPU", INTRO, show_hypothesis=False)

df = load_runs_or_stop()

# --- KPI ---------------------------------------------------------------------
st.subheader("Zbior danych w liczbach")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Przebiegi", len(df))
c2.metric("Maszyny", df["device_machine_name"].nunique() if "device_machine_name" in df.columns else "—")
c3.metric("Urzadzenia", df["device_short"].nunique() if "device_short" in df.columns else "—")
c4.metric("Zadania", df["task"].nunique())
if "started_at" in df.columns and df["started_at"].notna().any():
    lo, hi = df["started_at"].min(), df["started_at"].max()
    c5.metric("Zakres dat", f"{lo:%Y-%m-%d} – {hi:%Y-%m-%d}")
else:
    c5.metric("Zakres dat", "—")

if "arch_class" in df.columns:
    st.caption("Podzial urzadzen wg architektury (wymiar grupujacy w hipotezie - NPU to trzecia architektura obok CPU i GPU):")
    counts = df.drop_duplicates("device_short")["arch_class"].value_counts()
    arch_cols = st.columns(max(len(counts), 1))
    for col, (arch, n) in zip(arch_cols, counts.items()):
        col.metric(arch, int(n))

growing_note()
st.divider()

# --- Legenda kolorow ----------------------------------------------------------
st.subheader("Legenda kolorow (to samo kodowanie na kazdym wykresie)")
render_device_legend(sorted(df["device_device_type"].dropna().unique()) if "device_device_type" in df.columns else None)
st.divider()

# --- device_only vs whole_system --------------------------------------------
st.subheader("Co dokladnie mierzy pomiar energii")
st.markdown(SCOPE_EXPLAINER)
st.divider()

# --- Heatmapa pokrycia -----------------------------------------------------
st.subheader("Pokrycie eksperymentu: architektura x zadanie x faza")
matrix = coverage_matrix(df, index="device_type_label", columns=["task", "phase"])
fig = coverage_heatmap(matrix, title="Liczba przebiegow w kazdej kombinacji",
                       x_title="zadanie / faza", y_title="typ urzadzenia")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
render_figure_export_buttons(fig, ["pokrycie_eksperymentu"], key_prefix="coverage")
interpretation(
    co_pokazuje=(
        "Ile przebiegow zebrano dla kazdej pary *typ urzadzenia x (zadanie / faza)*. "
        "Ciemniejsza komorka = wiecej danych; komorka pusta (0) = tej kombinacji jeszcze nie ma."
    ),
    jak_czytac=(
        "Szukaj bialych/jasnych pol - to luki w macierzy eksperymentu. Faza `train` wystepuje "
        "tylko dla CPU / GPU NVIDIA / GPU AMD (ROCm); iGPU i NPU maja z zalozenia tylko "
        "`inference`. NPU i iGPU Intel maja waska kolumne (tylko klasyfikacja obrazow)."
    ),
    co_wynika=(
        "Porownania miedzy architekturami sa mocne tam, gdzie kilka wierszy ma ciemne komorki "
        "w tej samej kolumnie (np. klasyfikacja obrazow / inference). Tam, gdzie tylko jedna "
        "architektura ma dane, wniosku miedzy-architekturowego nie da sie postawic - te miejsca "
        "sa oznaczone na kolejnych stronach jako `n` male / brak danych."
    ),
    hipoteza=(
        "Rdzen testu hipotezy (CPU vs GPU przy malym batchu) rozgrywa sie w wierszach CPU / "
        "GPU NVIDIA / GPU AMD dla klasyfikacji obrazow i analizy wydzwieku - tam danych jest "
        "najwiecej."
    ),
)

st.divider()
st.subheader("Ostatnie przebiegi")
recent_cols = [c for c in ["started_at", "device_machine_name", "device_short", "arch_class",
                           "task", "model_name", "phase", "precision", "batch_size",
                           "duration_s", "energy_per_sample_j"] if c in df.columns]
table_block(
    df.sort_values("started_at", ascending=False)[recent_cols].head(25),
    name_parts=["ostatnie_przebiegi"],
    key="recent",
    caption="Pelny eksport calego zbioru - zakladka 10. Eksport danych.",
    height=420,
)
