"""Strona 3 - Kompromis wydajnosc <-> energia (front Pareto). Wprost: "szybciej != oszczedniej"."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import pareto_scatter
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.data_access.aggregation import agg_two_metrics, list_tasks, pareto_front
from dashboard.i18n.pl import TASK_LABELS, label, phase_label

INTRO = """
Kazdy punkt to jedna konfiguracja (architektura x model x batch size x precyzja): jej
**przepustowosc** (os X, wiecej = szybciej) i **energia na probke** (os Y, nizej =
oszczedniej). Obie osie logarytmiczne. Linia przerywana laczy **front Pareto** -
konfiguracje, ktorych nie da sie poprawic na jednej osi bez pogorszenia drugiej.
"""

header("Kompromis wydajnosc <-> energia (Pareto)", INTRO)
df = load_runs_or_stop()
df = df[df["task"] != "llm_inference"]

c1, c2 = st.columns(2)
with c1:
    phases = sorted(df["phase"].dropna().unique())
    phase = st.radio("Faza", phases, index=phases.index("inference") if "inference" in phases else 0,
                     horizontal=True, format_func=phase_label)
with c2:
    prec_opts = ["(wszystkie)"] + sorted(df["precision"].dropna().unique())
    precision = st.radio("Precyzja", prec_opts, horizontal=True)

render_device_legend(sorted(df["device_device_type"].dropna().unique()))

scope = df[df["phase"] == phase]
if precision != "(wszystkie)":
    scope = scope[scope["precision"] == precision]
if scope.empty:
    st.warning("Brak przebiegow dla wybranych filtrow.")
    st.stop()

by = ["device_device_type", "device_type_label", "device_short", "arch_class",
      "task", "model_name", "batch_size", "precision"]
points = agg_two_metrics(scope, "throughput_samples_per_s", "energy_per_sample_j", by=by)
points = points[points["throughput_samples_per_s"].notna() & points["energy_per_sample_j"].notna()]
if points.empty:
    st.warning("Brak kompletnych par (przepustowosc, energia/probke) dla wybranych filtrow.")
    st.stop()
points["n_runs"] = points["n_runs"].fillna(points["n"])

fronts = []
for _, g in points.groupby("task", observed=True):
    fronts.append(pareto_front(g, "throughput_samples_per_s", "energy_per_sample_j",
                               x_better="higher", y_better="lower"))
points["on_front"] = pd.concat(fronts).reindex(points.index).fillna(False)

fig = pareto_scatter(points, facet_col="task", front_col="on_front",
                     title=f"Front Pareto — faza {phase_label(phase)}"
                           + (f", {precision}" if precision != "(wszystkie)" else ""))
figure_block(
    fig,
    name_parts=["pareto", phase, None if precision == "(wszystkie)" else precision],
    key="pareto",
    co_pokazuje=(
        "Rozklad wszystkich konfiguracji w przestrzeni *przepustowosc x energia/probke*, panel na "
        "zadanie. Kolor + ksztalt = architektura (na wykresie punktowym sama barwa moze nie "
        "wystarczyc dla 6 serii, stad dodatkowo ksztalt). Linia przerywana = front Pareto."
    ),
    jak_czytac=(
        "Lewy-dolny rog = malo energii i... malo szybko; prawy-dolny = szybko **i** oszczednie "
        "(idealnie). Punkty na froncie to najlepsze mozliwe kompromisy. Punkty daleko nad frontem "
        "sa zdominowane - istnieje konfiguracja szybsza i oszczedniejsza zarazem. Serie iGPU/NPU to "
        "`whole_system` - ich pozycja na osi Y jest zawyzona zakresem pomiaru."
    ),
    co_wynika=(
        "Front tworza niemal wylacznie GPU dyskretne i NPU (prawy-dolny). Punkty CPU leza **nad** "
        "frontem - dla kazdej konfiguracji CPU istnieje konfiguracja GPU jednoczesnie szybsza i "
        "oszczedniejsza. Punkty CPU przy batch=1 sa najblizej frontu, ale go nie osiagaja."
    ),
    hipoteza=(
        "Pareto **osłabia** hipoteze w mocnej formie: zaden punkt CPU nie wchodzi na front. "
        "Najblizej sa punkty CPU dla lekkiego modelu przy batch=1 - to jedyny slad kierunkowego "
        "poparcia hipotezy."
    ),
)

st.subheader("Konfiguracje na froncie Pareto")
front_tbl = points[points["on_front"]].sort_values(["task", "energy_per_sample_j"])
table_block(
    front_tbl[[c for c in ["task", "device_short", "arch_class", "model_name", "batch_size", "precision",
                           "throughput_samples_per_s", "energy_per_sample_j", "n"] if c in front_tbl.columns]],
    name_parts=["pareto_front", phase], key="front_tbl",
    caption="Konfiguracje niezdominowane - najlepsze kompromisy szybkosc/energia w kazdym zadaniu.",
)
growing_note()
