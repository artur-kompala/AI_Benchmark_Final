"""Strona 8 - Co dokladnie mierzymy i jak powtarzalne sa wyniki. Nie chodzi o straszenie
liczba rozbieznosci - chodzi o pokazanie, ze rozne zrodla mierza rozny ZAKRES, a realnym
sygnalem jakosci jest powtarzalnosc powtorzen (CV)."""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import empty_figure, hbar_metric_by_device, scatter_over_time
from dashboard.components.export_widgets import render_figure_export_buttons
from dashboard.components.narrative import interpretation
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.components.theme import (
    INK_MUTED,
    SURFACE,
    add_identity_line,
    apply_house_style,
    device_color_map,
    ordered_device_types,
)
from dashboard.data_access.aggregation import agg_metric
from dashboard.data_access.queries import load_run_stats_summary_df_cached
from dashboard.data_access.supabase_client import SupabaseConfigError
from dashboard.i18n.pl import DEVICE_TYPE_LABELS, MEASUREMENT_SCOPE_LABELS, SCOPE_EXPLAINER, label

INTRO = """
Trzy pytania: (1) **co** mierzy kazde zrodlo energii, (2) jak bardzo **powtarzalne** sa
przebiegi, (3) czy nie ma artefaktow. Rozbieznosc miedzy pomiarem programowym a
watomierzem to **glownie roznica zakresu** (gniazdko vs uklad), nie blad.
"""

header("Co mierzymy / powtarzalnosc", INTRO)
df = load_runs_or_stop()
render_device_legend(sorted(df["device_device_type"].dropna().unique()))

st.markdown(SCOPE_EXPLAINER)
st.divider()

# --- 1. software vs watomierz ------------------------------------------------
st.subheader("Pomiar programowy vs watomierz fizyczny (Tuya)")
both = df[(df["summary_energy_joules_software"] > 0) & (df["summary_energy_joules_smart_plug"] > 0)].copy()
if both.empty:
    st.info("Brak przebiegow z jednoczesnym pomiarem programowym i watomierzem.")
else:
    both["device_type_label"] = both["device_device_type"].map(lambda t: DEVICE_TYPE_LABELS.get(t, t))
    torder = ordered_device_types(both["device_device_type"].dropna().unique().tolist())
    cmap = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(torder).items()}
    fig_sp = px.scatter(both, x="summary_energy_joules_software", y="summary_energy_joules_smart_plug",
                        color="device_type_label", color_discrete_map=cmap, log_x=True, log_y=True,
                        category_orders={"device_type_label": [DEVICE_TYPE_LABELS.get(t, t) for t in torder]},
                        custom_data=["device_short", "task", "model_name", "batch_size"],
                        title="Energia: pomiar programowy (X) vs watomierz z gniazdka (Y)")
    fig_sp.update_traces(marker=dict(size=8, line=dict(color=SURFACE, width=1)), opacity=0.75)
    vals = pd.concat([both["summary_energy_joules_software"], both["summary_energy_joules_smart_plug"]])
    lo, hi = float(vals.min()), float(vals.max())
    add_identity_line(fig_sp, lo, hi, text="oba zrodla rowne")
    apply_house_style(fig_sp, title=fig_sp.layout.title.text, x_title=label("summary_energy_joules_software"),
                      y_title=label("summary_energy_joules_smart_plug"), log_x=True, log_y=True,
                      legend_title=label("device_type"))
    # jawny zakres w log10 - autorange plotly.js na tych danych potrafi rozjechac os
    rng = [math.log10(lo * 0.7), math.log10(hi * 1.4)]
    fig_sp.update_xaxes(range=rng)
    fig_sp.update_yaxes(range=rng)
    figure_block(
        fig_sp, name_parts=["pomiar_software_vs_watomierz"], key="sp",
        co_pokazuje="Kazdy punkt to jeden przebieg z dwoma pomiarami energii: programowym (sam uklad, os X) "
        "i z watomierza (caly komputer z gniazdka, os Y). Linia kropkowana = oba zrodla rowne.",
        jak_czytac="Punkty **nad** linia = watomierz pokazuje wiecej niz pomiar programowy. To oczekiwane: "
        "gniazdko obejmuje zasilacz, ekran, plyte, straty - a pomiar programowy tylko CPU/GPU.",
        co_wynika="Systematyczne, przewidywalne przesuniecie w gore - nie losowy rozrzut. To potwierdza, ze "
        "rozbieznosc wynika z ROZNEGO ZAKRESU pomiaru, nie z bledu ktoregokolwiek zrodla.",
        hipoteza="Do porownan miedzy architekturami uzywamy konsekwentnie pomiaru programowego (`device_only`) "
        "tam, gdzie jest - zeby zakres byl porownywalny. Watomierz sluzy do kontroli, nie do rankingu.",
    )

# --- 2. power_discrepancy_pct ---------------------------------------------------
st.subheader("Rozbieznosc zrodel pomiaru wg architektury")
disc = df[df["summary_power_discrepancy_pct"].notna()]
if disc.empty:
    st.info("Brak zapisanej power_discrepancy_pct (liczona tylko tam, gdzie byl watomierz).")
else:
    agg_disc = agg_metric(disc, "summary_power_discrepancy_pct",
                          by=["device_device_type", "device_type_label", "arch_class"], per_group_first=False)
    fig_disc = hbar_metric_by_device(agg_disc, value_col="mean", error_col="std",
                                     x_title=label("summary_power_discrepancy_pct"),
                                     title="Srednia rozbieznosc watomierz vs pomiar programowy")
    figure_block(
        fig_disc, name_parts=["pomiar_rozbieznosc"], key="disc",
        co_pokazuje="Srednia (± odch. std.) rozbieznosci [%] miedzy watomierzem a pomiarem programowym, "
        "per architektura. Dane sa tylko dla CPU i GPU AMD (ROCm) - tam byl podpiety watomierz.",
        jak_czytac="Wartosci rzedu kilkudziesieciu procent to normalny efekt tego, ze watomierz mierzy caly "
        "komputer. Nie czytaj tego jako 'blad pomiaru' - to roznica zakresu.",
        co_wynika="Rozbieznosc jest duza, ale systematyczna i podobna miedzy architekturami, wiec nie "
        "zaburza porownania (uzywamy jednego, spojnego zrodla). Prawdziwy sygnal jakosci to CV nizej.",
    )

# --- 3. CV energii (glowny sygnal jakosci) ---------------------------------
st.subheader("Powtarzalnosc: wspolczynnik zmiennosci energii (CV)")
try:
    stats = load_run_stats_summary_df_cached()
except SupabaseConfigError as exc:
    st.error(str(exc)); stats = None
except Exception:
    st.info("Nie udalo sie wczytac run_stats_summary - prawdopodobnie migracja "
            "db/migrations/0004_repetitions_stats_quantization.sql nie zostala uruchomiona.")
    stats = None

if stats is not None and not stats.empty and "coefficient_of_variation_energy" in stats.columns:
    cv = stats[stats["coefficient_of_variation_energy"].notna() & stats["task"].ne("llm_inference")].copy()
    cv["device_type_label"] = cv["device_device_type"].map(lambda t: DEVICE_TYPE_LABELS.get(t, t))
    agg_cv = agg_metric(cv, "coefficient_of_variation_energy",
                        by=["device_device_type", "device_type_label", "task"], per_group_first=False)
    cvorder = ordered_device_types(cv["device_device_type"].dropna().unique().tolist())
    torder = [DEVICE_TYPE_LABELS.get(t, t) for t in cvorder]
    fig_cv = px.bar(agg_cv, y="device_type_label", x="mean", error_x="std", facet_col="task",
                    color="device_type_label", orientation="h",
                    color_discrete_map={DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(cvorder).items()},
                    category_orders={"device_type_label": torder[::-1]},
                    custom_data=["n"], title="Sredni CV energii calkowitej per architektura i zadanie")
    fig_cv.update_traces(marker_line_width=0, error_x=dict(color="#c3c2b7", thickness=1.2, width=4))
    fig_cv.add_vline(x=0.1, line=dict(color=INK_MUTED, width=1, dash="dot"),
                     annotation_text="CV = 0,10", annotation_position="top",
                     annotation_font=dict(size=10, color=INK_MUTED))
    apply_house_style(fig_cv, title=fig_cv.layout.title.text, x_title=label("coefficient_of_variation_energy"),
                      y_title="", legend_title=label("device_type"), show_legend=False)
    figure_block(
        fig_cv, name_parts=["pomiar_cv_energii"], key="cv",
        co_pokazuje="Sredni **wspolczynnik zmiennosci** energii calkowitej (std/mean po powtorzeniach tej "
        "samej konfiguracji), per architektura i zadanie. Linia = CV 0,10 (10%).",
        jak_czytac="Nizej = wyniki lepiej sie powtarzaja. CV < 0,10 to bardzo dobra powtarzalnosc; "
        "CV > 0,30 oznacza, ze pojedynczy pomiar tej konfiguracji trzeba traktowac ostroznie.",
        co_wynika="Wiekszosc konfiguracji ma niski CV - agregaty na poprzednich stronach sa wiarygodne. "
        "Ogon wysokiego CV dotyczy glownie krotkich przebiegow (maly batch, malo iteracji).",
        hipoteza="Wysoka powtarzalnosc w kluczowych scenariuszach (klasyfikacja obrazow, male batche) "
        "oznacza, ze wnioski z tabeli decyzyjnej (str. 9) nie sa artefaktem szumu.",
    )

# --- 4. probki mocy vs prog ------------------------------------------------
st.subheader("Liczba probek mocy per przebieg")
fig_ps = scatter_over_time(df, y_col="summary_power_samples_count", y_title=label("summary_power_samples_count"),
                           title="Liczba probek mocy zebranych w oknie pomiaru", ref_y=20,
                           ref_text="prog wiarygodnosci (20 probek)")
figure_block(
    fig_ps, name_parts=["pomiar_probki_mocy"], key="ps",
    co_pokazuje="Ile probek mocy zebrano w kazdym przebiegu (w czasie). Ponizej ~20 probek srednia/szczyt "
    "mocy staja sie praktycznie pojedynczym odczytem, nie rzetelna srednia.",
    jak_czytac="Punkty pod czerwona linia = mniej wiarygodne avg/peak power dla tego przebiegu. Nowsze "
    "przebiegi maja tego mniej (runner skaluje teraz liczbe iteracji z wyprzedzeniem).",
    co_wynika="Krotkie przebiegi (batch=1) najczesciej wpadaja pod prog - dlatego moc chwilowa (str. 6) "
    "czytamy jako rozklad, nie pojedyncze wartosci, a energie liczymy z powtorzen.",
)

# --- 5. measurement_scope ------------------------------------------------------
st.subheader("Rozklad zakresu pomiaru (measurement_scope)")
if "measurement_scope" in df.columns:
    sc = df["measurement_scope"].map(lambda s: MEASUREMENT_SCOPE_LABELS.get(s, s)).value_counts().reset_index()
    sc.columns = ["zakres", "liczba_przebiegow"]
    fig_sc = px.bar(sc, x="liczba_przebiegow", y="zakres", orientation="h", title="Ile przebiegow ma jaki zakres pomiaru")
    fig_sc.update_traces(marker_color="#2a78d6", marker_line_width=0)
    apply_house_style(fig_sc, title=fig_sc.layout.title.text, x_title="liczba przebiegow", y_title="", show_legend=False)
    figure_block(
        fig_sc, name_parts=["pomiar_scope"], key="scope",
        co_pokazuje="Ile przebiegow zmierzono jako `device_only` (sam uklad) a ile jako `whole_system` "
        "(caly SoC/komputer). NPU/iGPU zawsze `whole_system` - nie ma dla nich licznika samego ukladu.",
        jak_czytac="To nie jest metryka jakosci - to informacja, ktorych serii NIE wolno zestawiac wprost "
        "na osi J/probke bez zastrzezenia (na wykresach: szrafura).",
        co_wynika="Wiekszosc danych to `device_only` (CPU/CUDA/ROCm) - rdzen porownania CPU vs GPU jest "
        "w spojnym zakresie. iGPU/NPU trzymamy w osobnej interpretacji.",
    )

# --- 6. flops_per_watt anomalia --------------------------------------------
st.subheader("Anomalia: FLOPS/W dla ROCm")
fpw = df[df["summary_flops_per_watt"].notna()]
if not fpw.empty:
    agg_fpw = agg_metric(fpw, "summary_flops_per_watt",
                         by=["device_device_type", "device_type_label", "arch_class"], per_group_first=False)
    fig_fpw = hbar_metric_by_device(agg_fpw, value_col="mean", x_title=label("summary_flops_per_watt"),
                                    title="Srednia wydajnosc obliczeniowa FLOPS/W per architektura")
    fig_fpw.update_xaxes(type="log")
    figure_block(
        fig_fpw, name_parts=["pomiar_flops_per_watt"], key="fpw",
        co_pokazuje="Srednia FLOPS/W per architektura (skala logarytmiczna). FLOPS sa SZACOWANE z profilu "
        "modelu, nie mierzone bezposrednio.",
        jak_czytac="Spodziewamy sie, ze GPU dyskretne (CUDA/ROCm) beda wysoko. ROCm wypada tu jednak "
        "systematycznie **nizej** niz CUDA i iGPU przy podobnej pracy.",
        co_wynika="To prawdopodobnie **artefakt szacowania** FLOPS lub mocy dla ROCm (inny licznik mocy, "
        "cala karta desktopowa vs laptopowe GPU), a nie realna 4-5x nizsza efektywnosc. Traktujemy jako "
        "'do weryfikacji' i nie wyciagamy z FLOPS/W mocnych wnioskow - metryka rozstrzygajaca to J/probke.",
    )

growing_note()
