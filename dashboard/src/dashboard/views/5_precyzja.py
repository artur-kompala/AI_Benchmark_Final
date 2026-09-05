"""Strona 5 - Wplyw precyzji (fp32 / fp16 / int8). Energia i dokladnosc obok siebie -
nizsza energia nie jest argumentem, jesli towarzyszy jej spadek accuracy. Na dole:
mini-sekcja eksploracyjna dla llm_inference (poza glownymi porownaniami)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.charts import grouped_bar_generic
from dashboard.components.export_widgets import render_figure_export_buttons
from dashboard.components.narrative import interpretation
from dashboard.components.page import growing_note, header, load_runs_or_stop, table_block
from dashboard.components.theme import SURFACE, apply_house_style
from dashboard.data_access.aggregation import agg_metric
from dashboard.i18n.pl import (
    DEVICE_TYPE_LABELS,
    DEVICE_TYPE_ORDER,
    PRECISION_COLORS,
    PRECISION_ORDER,
    QUANTIZATION_STATUS_LABELS,
    TASK_LABELS,
    label,
    phase_label,
)

INTRO = """
Precyzja obliczen to podstawowa dzwignia energooszczednosci opisana w pracy (par. 1.4):
przejscie fp32 -> int8 obniza koszt pamieci i pozwala uzyc szybszych jednostek. Ale liczy
sie tez **dokladnosc** - dlatego energia i accuracy sa tu zawsze obok siebie.

**Ograniczenia z danych:** `int8` (dynamiczna kwantyzacja PyTorch) uruchamiano praktycznie
tylko na CPU - to celowe i zgodne z praca. `iGPU Intel (OpenVINO)` i `NPU` maja tylko `fp32`.
"""

header("Wplyw precyzji (fp32 / fp16 / int8)", INTRO)
df = load_runs_or_stop()
df_main = df[df["task"] != "llm_inference"]

tasks = sorted(df_main["task"].dropna().unique())
c1, c2 = st.columns(2)
with c1:
    task = st.selectbox("Zadanie", tasks, format_func=lambda t: TASK_LABELS.get(t, t))
with c2:
    phases = sorted(df_main[df_main["task"] == task]["phase"].dropna().unique())
    phase = st.radio("Faza", phases, index=phases.index("inference") if "inference" in phases else 0,
                     horizontal=True, format_func=phase_label)

scope = df_main[(df_main["task"] == task) & (df_main["phase"] == phase)]
if scope.empty:
    st.warning("Brak przebiegow dla wybranych filtrow.")
    st.stop()

by = ["device_type_label", "device_device_type", "arch_class", "precision"]
prec_order = [p for p in PRECISION_ORDER if p in scope["precision"].unique()]
dev_order = [DEVICE_TYPE_LABELS.get(t, t) for t in DEVICE_TYPE_ORDER
             if t in scope["device_device_type"].unique()][::-1]
cat_orders = {"precision": prec_order, "device_type_label": dev_order}

energy = agg_metric(scope, "energy_per_sample_j", by=by)
acc = agg_metric(scope, "summary_accuracy", by=by, per_group_first=False)

col_e, col_a = st.columns(2)
with col_e:
    fig_e = grouped_bar_generic(
        energy, x_col="device_type_label", y_col="mean", color_col="precision",
        error_col="std", color_map=PRECISION_COLORS, category_orders=cat_orders,
        y_title=label("energy_per_sample_j"), log_y=True, orientation="h",
        title=f"Energia na probke — {TASK_LABELS.get(task, task)}",
    )
    st.plotly_chart(fig_e, width="stretch", config={"displayModeBar": False})
with col_a:
    fig_a = grouped_bar_generic(
        acc, x_col="device_type_label", y_col="mean", color_col="precision",
        error_col="std", color_map=PRECISION_COLORS, category_orders=cat_orders,
        y_title=label("summary_accuracy"), y_range=[0, 1], orientation="h",
        title=f"Dokladnosc (accuracy) — {TASK_LABELS.get(task, task)}",
    )
    st.plotly_chart(fig_a, width="stretch", config={"displayModeBar": False})

e1, e2 = st.columns(2)
with e1:
    render_figure_export_buttons(fig_e, ["precyzja_energia", task, phase], key_prefix="prec_e")
with e2:
    render_figure_export_buttons(fig_a, ["precyzja_accuracy", task, phase], key_prefix="prec_a")

interpretation(
    co_pokazuje=(
        "Po lewej: srednia energia na probke dla kazdej precyzji i architektury. Po prawej: srednia "
        "dokladnosc w tych samych przekrojach (skala 0-1). Slupki pogrupowane po precyzji "
        "(fp32 najciemniejszy -> int8 najjasniejszy)."
    ),
    jak_czytac=(
        "Czytaj oba wykresy razem: spadek slupka energii po lewej ma sens tylko, jesli slupek "
        "accuracy po prawej nie spada zauwazalnie. `int8` pojawia sie glownie przy CPU."
    ),
    co_wynika=(
        "Na CPU int8 zwykle obniza energie/probke wzgledem fp32, ale efekt bywa umiarkowany, bo "
        "kwantyzacja jest czesto tylko czesciowa (patrz tabela nizej). fp16 na CPU rzadko pomaga "
        "(brak natywnego wsparcia), na GPU - tak."
    ),
    hipoteza=(
        "Precyzja to druga - obok batch size - dzwignia z hipotezy. int8 na CPU moze dodatkowo "
        "poprawic pozycje CPU w porownaniu energetycznym z GPU (ktore w tym zbiorze int8 nie uzywa)."
    ),
)

st.subheader("Status kwantyzacji INT8")
q = scope[scope["precision"] == "int8"]
if not q.empty and "summary_quantization_status" in q.columns:
    qt = (q.assign(status=q["summary_quantization_status"].fillna("").map(lambda s: QUANTIZATION_STATUS_LABELS.get(s, s or "brak")))
            .groupby(["device_type_label", "model_name", "status"], observed=True).size().reset_index(name="liczba_przebiegow"))
    table_block(qt, name_parts=["precyzja_status_int8", task], key="qstat",
                caption="`partial` = skwantyzowano tylko czesc warstw (np. Linear) - energia odzwierciedla model "
                        "czesciowo int8, nie pelny int8. To wazne zastrzezenie przy czytaniu slupkow energii.")
else:
    st.info("Brak przebiegow int8 dla tego zadania/fazy.")

# --- mini-sekcja eksploracyjna: llm_inference -------------------------------
st.divider()
with st.expander("llm_inference (eksploracyjnie - poza glownymi porownaniami)", expanded=False):
    st.markdown(
        "`llm_inference` (tiny-gpt2, tylko CPU, batch=1) ma **skrajnie zmienna liczbe iteracji** "
        "(`num_iterations` od ~300 do ~5000) i brak realnych grup powtorzen, wiec nie wchodzi do "
        "glownych porownan ani do tabeli decyzyjnej. Ponizej sam poglad zaleznosci energii od "
        "dlugosci generacji."
    )
    llm = df[df["task"] == "llm_inference"].copy()
    if llm.empty:
        st.info("Brak przebiegow llm_inference.")
    else:
        fig_llm = px.scatter(
            llm, x="num_iterations", y="energy_per_sample_j", color="precision",
            color_discrete_map=PRECISION_COLORS, custom_data=["device_short", "duration_s"],
            title="llm_inference: energia/probke vs liczba iteracji (CPU, tiny-gpt2)",
        )
        fig_llm.update_traces(marker=dict(size=10, line=dict(color=SURFACE, width=1)))
        apply_house_style(fig_llm, title=fig_llm.layout.title.text, x_title="liczba iteracji (num_iterations)",
                          y_title=label("energy_per_sample_j"), legend_title=label("precision"))
        st.plotly_chart(fig_llm, width="stretch", config={"displayModeBar": False})
        render_figure_export_buttons(fig_llm, ["llm_inference_eksploracja"], key_prefix="llm")
        st.caption("Brak monotonicznego trendu i ogromny rozrzut - dlatego llm_inference jest tylko sygnalem "
                   "pomocniczym, nie podstawa wnioskow o architekturach.")

growing_note()
