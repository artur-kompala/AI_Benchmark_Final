"""Strona 1 - Efektywnosc energetyczna. Rdzen hipotezy: energia na probke [J/probke]
wg architektury, w panelach per zadanie."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import bar_metric_by_device
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.data_access.aggregation import agg_metric, list_tasks
from dashboard.i18n.pl import TASK_LABELS, label, phase_label

INTRO = """
Ile energii kosztuje przetworzenie **jednej probki** (obrazu / tekstu) na kazdej
architekturze. To jest metryka rozstrzygajaca hipoteze pracy. Ten sam wynik pokazujemy
dwojako: **J/probke** (nizej = lepiej, skala logarytmiczna) oraz **probki/dzul**
(wyzej = lepiej) - dla intuicji "ile dostaje sie za ten sam dzul".
"""

header("Efektywnosc energetyczna (energia na probke)", INTRO)
df = load_runs_or_stop()
df = df[df["task"] != "llm_inference"]  # llm_inference: osobna sekcja eksploracyjna (str. 5)

# --- filtry (jeden rzad nad wykresami) ---
tasks = list_tasks(df, order=list(TASK_LABELS))
c1, c2 = st.columns([2, 2])
with c1:
    precisions = sorted(df["precision"].dropna().unique())
    default_prec = "fp32" if "fp32" in precisions else precisions[0]
    precision = st.radio("Precyzja", precisions, index=precisions.index(default_prec), horizontal=True)
with c2:
    phases = sorted(df["phase"].dropna().unique())
    phase = st.radio("Faza", phases, index=phases.index("inference") if "inference" in phases else 0,
                     horizontal=True, format_func=phase_label)

render_device_legend(sorted(df["device_device_type"].dropna().unique()))

scope = df[(df["precision"] == precision) & (df["phase"] == phase)]
if scope.empty:
    st.warning(f"Brak przebiegow dla precyzji {precision} i fazy {phase_label(phase)}.")
    st.stop()

by = ["device_device_type", "device_type_label", "arch_class", "task"]
agg_eps = agg_metric(scope, "energy_per_sample_j", by=by)
agg_spj = agg_metric(scope, "samples_per_joule", by=by)

# --- wykres 1: J/probke ---
fig_eps = bar_metric_by_device(
    agg_eps, value_col="mean", error_col="std", facet_col="task",
    category_order=tasks, y_title=label("energy_per_sample_j"), log_y=True,
    title=f"Energia na probke wg architektury — {precision}, {phase_label(phase)}",
)
figure_block(
    fig_eps,
    name_parts=["efektywnosc_j_na_probke", precision, phase],
    key="eps",
    co_pokazuje=(
        f"Srednia **energia na probke** (J) dla precyzji `{precision}`, faza *{phase_label(phase)}*, "
        "osobny panel na zadanie. Slupek = srednia po grupach powtorzen; wasy = odchylenie "
        "standardowe miedzy konfiguracjami (rozne batch size / modele / maszyny w tej architekturze). "
        "Skala Y logarytmiczna - rozrzut siega ~100x."
    ),
    jak_czytac=(
        "Nizszy slupek = mniej energii na probke = **oszczedniej**. Slupki szrafowane to serie "
        "`whole_system` (iGPU / NPU) - mierza caly SoC / komputer, wiec sa z natury wyzej i nie "
        "porownuje sie ich wprost z `device_only` (CPU/GPU). `n` w tooltipie = liczba grup powtorzen."
    ),
    co_wynika=(
        "GPU dyskretne (CUDA) i NPU maja najnizsza energie/probke w kazdym zadaniu. CPU jest wyzej "
        "**zawsze** - od ok. 1,6x (lekki model, maly batch) do kilkunastu-kilkudziesieciu razy "
        "(ciezki model, trening). iGPU/NPU (szrafura) to `whole_system` - ich slupek jest zawyzony "
        "zakresem pomiaru, wiec realna przewaga nad CPU jest wieksza, niz widac."
    ),
    hipoteza=(
        "Ten wykres **nie potwierdza** hipotezy w mocnej formie: CPU nie dorownuje GPU energetycznie "
        "w zadnym zestawieniu srednich. Hipoteza trzyma sie co najwyzej kierunkowo - luka jest "
        "najmniejsza dla lekkiego modelu i malego batcha (str. 4). Czy to 'porownywalna efektywnosc', "
        "czy nadal wyrazna przewaga GPU - ocenia str. 9."
    ),
)

# --- wykres 2: probki/dzul ---
fig_spj = bar_metric_by_device(
    agg_spj, value_col="mean", error_col="std", facet_col="task",
    category_order=tasks, y_title=label("samples_per_joule"), log_y=True,
    title=f"Probki na dzul wg architektury — {precision}, {phase_label(phase)}",
)
figure_block(
    fig_spj,
    name_parts=["efektywnosc_probki_na_dzul", precision, phase],
    key="spj",
    co_pokazuje=(
        "Ten sam wynik odwrocony: ile **probek przypada na jeden dzul** energii "
        "(= 1 / energia-na-probke). Wyzej = lepiej."
    ),
    jak_czytac=(
        "Wysoki slupek = duzo pracy za maly koszt energetyczny. To ten sam porzadek co wyzej, "
        "tylko intuicyjnie \"wiecej = lepiej\". Skala logarytmiczna."
    ),
    co_wynika=(
        "Architektury z prawego konca (CUDA, NPU) dostaja kilkadziesiat-kilkaset probek/dzul; "
        "CPU zwykle kilka-kilkanascie. Odstep jest staly - CPU nie przeskakuje GPU w zadnym zadaniu."
    ),
)

st.subheader("Dane zrodlowe (agregaty)")
table_cols = ["task", "device_type_label", "arch_class", "mean", "std", "n", "n_runs", "small_n"]
table_block(
    agg_eps[[c for c in table_cols if c in agg_eps.columns]].rename(columns={"mean": "energy_per_sample_j", "std": "std_energy_per_sample_joules"}),
    name_parts=["efektywnosc_tabela", precision, phase],
    key="eps_tbl",
    caption="mean/std liczone dwustopniowo: najpierw srednia po repetition_group_id, potem po konfiguracjach w danej architekturze.",
)
growing_note()
