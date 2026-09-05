"""Strona 4 - Wplyw rozmiaru partii (batch size). Kluczowy test hipotezy: przy malym
batchu roznica energii CPU vs GPU maleje (albo znika)."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import line_metric_vs_batch
from dashboard.components.format import fmt_pl
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.data_access.aggregation import agg_metric, crossover_batches, list_tasks
from dashboard.i18n.pl import TASK_LABELS, label, phase_label

INTRO = """
Energia na probke i przepustowosc **w funkcji rozmiaru partii** (batch size), osobna
linia na architekture, panel na model. Hipoteza pracy wskazuje wlasnie male batche jako
scenariusz, w ktorym CPU dogania GPU energetycznie - tu widac, gdzie krzywe sie schodza
albo przecinaja.
"""

header("Wplyw rozmiaru partii (batch size)", INTRO)
df = load_runs_or_stop()
df = df[df["task"] != "llm_inference"]

tasks = list_tasks(df, order=list(TASK_LABELS))
c1, c2, c3 = st.columns(3)
with c1:
    task = st.selectbox("Zadanie", tasks, format_func=lambda t: TASK_LABELS.get(t, t))
with c2:
    phases = sorted(df[df["task"] == task]["phase"].dropna().unique())
    phase = st.radio("Faza", phases, index=phases.index("inference") if "inference" in phases else 0,
                     horizontal=True, format_func=phase_label)
with c3:
    precisions = sorted(df[df["task"] == task]["precision"].dropna().unique())
    precision = st.radio("Precyzja", precisions, index=precisions.index("fp32") if "fp32" in precisions else 0,
                         horizontal=True)

render_device_legend(sorted(df["device_device_type"].dropna().unique()))

scope = df[(df["task"] == task) & (df["phase"] == phase) & (df["precision"] == precision)]
if scope.empty:
    st.warning("Brak przebiegow dla wybranych filtrow.")
    st.stop()

by = ["device_device_type", "device_type_label", "arch_class", "model_name", "batch_size"]
eps = agg_metric(scope, "energy_per_sample_j", by=by)
thr = agg_metric(scope, "throughput_samples_per_s", by=by, per_group_first=False)

# przeciecia krzywej CPU z pozostalymi (energia/probke)
cross = []
for model, g in eps.groupby("model_name", observed=True):
    reshaped = g.rename(columns={"mean": "y"}).assign(x=g["batch_size"])
    for c in crossover_batches(reshaped, "x", "y", "device_device_type", reference="cpu"):
        c["model_name"] = model
        cross.append(c)

fig_eps = line_metric_vs_batch(
    eps, value_col="mean", error_col="sem", facet_col="model_name",
    y_title=label("energy_per_sample_j"), log_y=True,
    title=f"Energia na probke vs batch size — {TASK_LABELS.get(task, task)}, {precision}, {phase_label(phase)}",
    crossings=cross,
)
cross_series = sorted({c["seria"] for c in cross})
cross_txt = (
    "Krzywa CPU przecina: "
    + ", ".join(f"{s} (~batch {fmt_pl(min(c['x_przeciecia'] for c in cross if c['seria'] == s), 1)})"
                for s in cross_series)
    if cross else "W tym przekroju krzywa CPU nie przecina zadnej innej w mierzonym zakresie batch."
)
figure_block(
    fig_eps,
    name_parts=["batch_energia", task, precision, phase],
    key="eps",
    co_pokazuje=(
        f"Srednia **energia na probke** w funkcji batch size, linia na typ urzadzenia, panel na model. "
        f"Skala Y logarytmiczna. Pionowe kropkowane linie = przeciecia krzywej CPU z inna krzywa. {cross_txt}"
    ),
    jak_czytac=(
        "Krzywa opadajaca = wiekszy batch amortyzuje koszt staly, wiec energia/probke spada. Tam, "
        "gdzie krzywa CPU jest **ponizej** innej krzywej, CPU jest w tym punkcie oszczedniejszy. "
        "Linie kropkowane (iGPU/NPU) to `whole_system` - inny zakres pomiaru."
    ),
    co_wynika=(
        "Krzywa CPU najczesciej przecina sie z **ROCm** (desktopowa RX 7800 XT) - przy malych/srednich "
        "batchach CPU bywa tam nizej. Krzywe **CUDA** (laptopowe GPU) leza znacznie nizej niz CPU na "
        "calym zakresie - z nimi CPU sie nie przecina. Dla ciezszych modeli GPU wygrywa wszedzie."
    ),
    hipoteza=(
        "Kierunkowe poparcie hipotezy: przeciecia z ROCm pokazuja, ze przewymiarowane desktopowe GPU "
        "przy lekkim zadaniu i malym batchu bywa mniej oszczedne niz CPU tej samej maszyny. "
        "Wobec energooszczednych GPU laptopowych (CUDA) tego efektu nie ma."
    ),
)

fig_thr = line_metric_vs_batch(
    thr, value_col="mean", error_col="sem", facet_col="model_name",
    y_title=label("throughput_samples_per_s"), log_y=True,
    title=f"Przepustowosc vs batch size — {TASK_LABELS.get(task, task)}, {precision}, {phase_label(phase)}",
)
figure_block(
    fig_thr,
    name_parts=["batch_przepustowosc", task, precision, phase],
    key="thr",
    co_pokazuje="Przepustowosc (probki/s) w funkcji batch size, ten sam uklad paneli.",
    jak_czytac="Krzywa rosnaca = wiekszy batch = lepsze wykorzystanie rownoleglosci. GPU zwykle zyskuje "
    "na batchu duzo bardziej niz CPU (stromsza krzywa).",
    co_wynika="Rozjezdzanie sie krzywych przepustowosci z batchem to druga strona tego samego zjawiska: "
    "przy malym batchu GPU nie jest w pelni wykorzystane, wiec i jego przewaga energetyczna topnieje.",
)

st.subheader("Dane zrodlowe (agregaty energii/probke)")
table_block(
    eps[[c for c in ["model_name", "device_type_label", "arch_class", "batch_size", "mean", "std", "n", "small_n"] if c in eps.columns]],
    name_parts=["batch_tabela", task, precision, phase], key="eps_tbl",
)
growing_note()
