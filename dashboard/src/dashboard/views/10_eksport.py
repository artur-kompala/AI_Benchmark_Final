"""Strona 10 - Eksport danych. Pelny zbior (runs + devices + run_summary), statystyka
powtorzen (run_stats_summary) oraz tabela decyzyjna ze strony 9 - wszystko do CSV/Excel
z opisowa nazwa pliku. Kazdy wykres i tabela w innych zakladkach ma tez wlasny eksport."""

from __future__ import annotations

import streamlit as st

from dashboard.components.export_widgets import render_dataframe_export_buttons
from dashboard.components.page import header, load_runs_or_stop
from dashboard.data_access.aggregation import build_decision_table, same_machine_winners
from dashboard.data_access.queries import load_run_stats_summary_df_cached
from dashboard.data_access.supabase_client import SupabaseConfigError

INTRO = """
Pobranie calego zbioru naraz - do dalszej analizy statystycznej w pracy (R / SPSS / Excel).
Dashboard czyta **wylacznie** z Supabase na zywo; ten zbior rosnie w miare naplywania
wynikow z kolejnych maszyn.
"""

header("Eksport danych", INTRO, show_hypothesis=False)
df = load_runs_or_stop()

with st.expander("Filtry", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tasks = st.multiselect("Zadanie", sorted(df["task"].dropna().unique()))
    with c2:
        devs = st.multiselect("Urzadzenie", sorted(df["device_short"].dropna().unique()) if "device_short" in df.columns else [])
    with c3:
        machines = st.multiselect("Maszyna", sorted(df["device_machine_name"].dropna().unique()) if "device_machine_name" in df.columns else [])
    with c4:
        precisions = st.multiselect("Precyzja", sorted(df["precision"].dropna().unique()))

f = df.copy()
if tasks:
    f = f[f["task"].isin(tasks)]
if devs:
    f = f[f["device_short"].isin(devs)]
if machines:
    f = f[f["device_machine_name"].isin(machines)]
if precisions:
    f = f[f["precision"].isin(precisions)]

st.write(f"Liczba wierszy: **{len(f)}** z {len(df)}")
st.dataframe(f, width="stretch", height=420)
render_dataframe_export_buttons(f, ["wyniki_benchmarkow"] + tasks, key_prefix="export_runs")

st.divider()
st.subheader("Statystyka powtorzen (run_stats_summary)")
st.caption("mean / std / coefficient_of_variation liczone przez runner po zakonczeniu wszystkich powtorzen "
           "kazdej kombinacji parametrow (patrz docs/measurement_methodology.md).")
try:
    stats_df = load_run_stats_summary_df_cached()
except SupabaseConfigError as exc:
    st.error(str(exc))
    stats_df = None
except Exception:
    st.info("Nie udalo sie wczytac run_stats_summary - najprawdopodobniej migracja "
            "db/migrations/0004_repetitions_stats_quantization.sql nie zostala uruchomiona na tym "
            "projekcie Supabase (patrz docs/setup_supabase.md).")
    stats_df = None

if stats_df is not None:
    if stats_df.empty:
        st.info("Brak wpisow w run_stats_summary - zaden przebieg jeszcze nie zakonczyl calej grupy powtorzen.")
    else:
        st.dataframe(stats_df, width="stretch", height=340)
        render_dataframe_export_buttons(stats_df, ["statystyka_powtorzen"], key_prefix="export_stats")

st.divider()
st.subheader("Tabela decyzyjna — w obrebie maszyn (ze strony 9)")
st.caption("Dla kazdego scenariusza (na danej maszynie, min. 2 urzadzenia): zwyciezca po energii na "
           "probke + przewaga nad kolejnym. To jest wersja uczciwa (ten sam krzem). llm_inference wykluczone.")
smw = same_machine_winners(df, value_col="energy_per_sample_j")
if smw.empty:
    st.info("Za malo danych do porownan w obrebie maszyn.")
else:
    st.dataframe(smw, width="stretch", height=360)
    render_dataframe_export_buttons(smw, ["tabela_decyzyjna_ta_sama_maszyna"], key_prefix="export_smw")

st.divider()
st.subheader("Tabela decyzyjna — zbiorcza (per typ urzadzenia, po wszystkich maszynach)")
st.caption("Pomocnicza: usrednia po maszynach, wiec CUDA (2 energooszczedne laptopy) zwykle wygrywa. "
           "Do porownan miedzy-architekturowych lepsza jest wersja w obrebie maszyn wyzej.")
dt = build_decision_table(df, value_col="energy_per_sample_j")
if dt.empty:
    st.info("Za malo danych do zbudowania tabeli decyzyjnej.")
else:
    st.dataframe(dt, width="stretch", height=300)
    render_dataframe_export_buttons(dt, ["tabela_decyzyjna_zbiorcza"], key_prefix="export_dt")
