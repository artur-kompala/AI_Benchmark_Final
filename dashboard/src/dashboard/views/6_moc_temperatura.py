"""Strona 6 - Moc chwilowa i temperatura. Energia = moc x czas, wiec rozklad mocy
chwilowej tlumaczy, skad biora sie roznice w energii sumarycznej."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.charts import box_metric_by_device
from dashboard.components.export_widgets import render_figure_export_buttons
from dashboard.components.narrative import interpretation
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.components.theme import (
    SURFACE,
    add_identity_line,
    apply_house_style,
    device_color_map,
    ordered_device_types,
)
from dashboard.data_access.aggregation import agg_metric
from dashboard.i18n.pl import DEVICE_TYPE_LABELS, TASK_LABELS, label, phase_label

INTRO = """
Rozklad **mocy chwilowej** (sredniej i szczytowej) na kazdej architekturze oraz - gdzie
dostepna - temperatura. To ogniwo posrednie hipotezy: *czas x moc chwilowa = energia
sumaryczna*. Architektura moze byc wolniejsza, ale jesli utrzymuje niska moc, i tak
wypadnie oszczedniej.
"""

header("Moc chwilowa i temperatura", INTRO)
df = load_runs_or_stop()
df = df[df["task"] != "llm_inference"]
render_device_legend(sorted(df["device_device_type"].dropna().unique()))

phase = st.radio("Faza", sorted(df["phase"].dropna().unique()),
                 index=0, horizontal=True, format_func=phase_label)
scope = df[df["phase"] == phase]

fig_avg = box_metric_by_device(scope, y_col="summary_avg_power_watts", y_title=label("summary_avg_power_watts"),
                               facet_col="task", title=f"Srednia moc chwilowa — {phase_label(phase)}")
figure_block(
    fig_avg, name_parts=["moc_srednia", phase], key="avg",
    co_pokazuje="Rozklad **sredniej mocy chwilowej** (W) per architektura, panel na zadanie. Pudelko = "
    "kwartyle, linia = mediana, krzyzyk = srednia, punkty = wartosci odstajace.",
    jak_czytac="Nizej = mniejszy pobor w trakcie pracy. Uwaga na zakres pomiaru: CPU/CUDA/ROCm to sam "
    "uklad (`device_only`), iGPU/NPU to caly SoC/komputer (`whole_system`) - te drugie sa z natury wyzej.",
    co_wynika="Laptopowe GPU NVIDIA (CUDA) czesto maja NIZSZA srednia moc niz CPU tej samej maszyny, mimo "
    "wyzszej wydajnosci - to wlasnie dlatego wygrywaja energetycznie. ROCm (desktopowa RX 7800 XT) "
    "trzyma duzo wyzsza moc.",
    hipoteza="Tam, gdzie CPU ma podobna lub nizsza moc chwilowa niz GPU (male obciazenie, maly batch), a "
    "czas nie jest drastycznie dluzszy - energia sumaryczna CPU dorownuje GPU. To mechanizm hipotezy.",
)

fig_peak = box_metric_by_device(scope, y_col="summary_peak_power_watts", y_title=label("summary_peak_power_watts"),
                                facet_col="task", title=f"Szczytowa moc chwilowa — {phase_label(phase)}")
figure_block(
    fig_peak, name_parts=["moc_szczytowa", phase], key="peak",
    co_pokazuje="To samo dla **mocy szczytowej** (W) - najwyzszy chwilowy pobor w oknie pomiaru.",
    jak_czytac="Wysokie szczyty przy niskiej sredniej = praca 'zrywami' (typowe dla malego batcha na CPU).",
    co_wynika="Szczyt istotny dla projektowania zasilania/chlodzenia, ale o energii sumarycznej decyduje "
    "srednia. Duza roznica peak-avg oznacza, ze uklad rzadko byl w pelni obciazony.",
)

# peak vs avg scatter
pa = scope[scope["summary_avg_power_watts"].notna() & scope["summary_peak_power_watts"].notna()].copy()
if not pa.empty:
    pa["device_type_label"] = pa["device_device_type"].map(lambda t: DEVICE_TYPE_LABELS.get(t, t))
    type_order = ordered_device_types(pa["device_device_type"].dropna().unique().tolist())
    cmap = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}
    fig_pa = px.scatter(pa, x="summary_avg_power_watts", y="summary_peak_power_watts",
                        color="device_type_label", color_discrete_map=cmap,
                        category_orders={"device_type_label": [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]},
                        custom_data=["device_short", "task", "batch_size"],
                        title=f"Szczyt vs srednia mocy — {phase_label(phase)}")
    fig_pa.update_traces(marker=dict(size=8, line=dict(color=SURFACE, width=1)), opacity=0.75)
    lo = float(min(pa["summary_avg_power_watts"].min(), pa["summary_peak_power_watts"].min()))
    hi = float(max(pa["summary_avg_power_watts"].max(), pa["summary_peak_power_watts"].max()))
    add_identity_line(fig_pa, lo, hi, text="szczyt = srednia")
    apply_house_style(fig_pa, title=fig_pa.layout.title.text, x_title=label("summary_avg_power_watts"),
                      y_title=label("summary_peak_power_watts"), legend_title=label("device_type"))
    figure_block(
        fig_pa, name_parts=["moc_szczyt_vs_srednia", phase], key="pa",
        co_pokazuje="Kazdy punkt to jeden przebieg: srednia moc (X) vs szczytowa (Y). Linia kropkowana to "
        "szczyt = srednia (praca w stalym obciazeniu).",
        jak_czytac="Punkty wysoko nad linia = duze zrywy przy niskim sredni obciazeniu. Blisko linii = uklad "
        "stale obciazony.",
        co_wynika="Przebiegi CPU przy batch=1 zwykle leza wysoko nad linia (praca zrywami) - to znak, ze CPU "
        "nie jest waskim gardlem i jest miejsce na efektywnosc energetyczna.",
    )

# temperatura, gdzie jest
temp = scope[scope["avg_temperature_c"].notna()]
st.subheader("Temperatura (tam, gdzie zmierzona)")
if temp.empty:
    st.info("Brak zapisanej avg_temperature_c dla wybranej fazy.")
else:
    fig_t = box_metric_by_device(temp, y_col="avg_temperature_c", y_title=label("avg_temperature_c"),
                                 title=f"Srednia temperatura — {phase_label(phase)}")
    figure_block(
        fig_t, name_parts=["temperatura", phase], key="temp",
        co_pokazuje="Rozklad sredniej temperatury (°C) per architektura - tylko przebiegi z zapisanym odczytem.",
        jak_czytac="Wyzsza temperatura zwykle idzie w parze z wyzsza moca chwilowa i dluzszym obciazeniem.",
        co_wynika="Temperatura to sygnal pomocniczy (nie wchodzi do hipotezy wprost), ale potwierdza wzorzec "
        "poboru mocy: goretsze uklady to te trzymajace wysoka moc przez caly przebieg.",
    )
    st.caption(f"Odczyt temperatury dostepny dla {temp['device_short'].nunique()} urzadzen "
               f"({len(temp)} z {len(scope)} przebiegow tej fazy).")

growing_note()
