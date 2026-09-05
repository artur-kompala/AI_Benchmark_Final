"""Rusztowanie strony - wspolne dla wszystkich 11 zakladek, zeby kazda strona miala ten
sam uklad i zeby KAZDY wykres/tabela mial(a) komplet: eksport (PNG/SVG/CSV/Excel, patrz
components/export*.py - mechanizmu NIE ruszamy) + blok narracji
(Co pokazuje / Jak czytac / Co z tego wynika)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.export_widgets import (
    render_dataframe_export_buttons,
    render_figure_export_buttons,
)
from dashboard.components.narrative import interpretation
from dashboard.data_access.queries import load_runs_df_cached
from dashboard.data_access.supabase_client import SupabaseConfigError
from dashboard.i18n.pl import (
    DATASET_GROWING_NOTE,
    DEVICE_TYPE_LABELS,
    DEVICE_TYPE_ORDER,
    DEVICE_TYPE_SCOPE,
    HYPOTHESIS_SHORT,
    label,
)
from dashboard.i18n.pl import DEVICE_TYPE_COLORS as _DT_COLORS


def load_runs_or_stop() -> pd.DataFrame:
    """Wczytuje przebiegi z Supabase (cache ttl=60). Zdegradowany, czytelny stan zamiast
    wyjatku przy braku konfiguracji / braku danych."""
    try:
        df = load_runs_df_cached()
    except SupabaseConfigError as exc:
        st.error(str(exc))
        st.stop()
    if df.empty:
        st.warning(
            "Brak danych w Supabase. Uruchom benchmark_runner na jednej z maszyn testowych, "
            "zeby wygenerowac wyniki - dashboard pokaze je automatycznie po odswiezeniu."
        )
        st.stop()
    return df


def header(title: str, intro: str, *, show_hypothesis: bool = True) -> None:
    st.title(title)
    if show_hypothesis:
        st.caption(HYPOTHESIS_SHORT)
    st.markdown(intro)
    st.divider()


def growing_note() -> None:
    st.caption(DATASET_GROWING_NOTE)


def render_device_legend(present_types: list[str] | None = None) -> None:
    """Legenda kolorow `device_type` - to samo kodowanie na kazdym wykresie w dashboardzie.
    Serie oznaczone (*) maja pomiar `whole_system` (caly SoC / komputer)."""
    order = [d for d in DEVICE_TYPE_ORDER if (present_types is None or d in present_types)]
    chips = []
    for dt in order:
        color = _DT_COLORS.get(dt, "#898781")
        star = " *" if DEVICE_TYPE_SCOPE.get(dt) == "whole_system" else ""
        chips.append(
            f"<span style='display:inline-block;margin:2px 10px 2px 0;white-space:nowrap;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
            f"background:{color};vertical-align:middle;margin-right:6px;'></span>"
            f"<span style='vertical-align:middle;font-size:0.9rem;'>{DEVICE_TYPE_LABELS.get(dt, dt)}{star}</span></span>"
        )
    st.markdown("".join(chips), unsafe_allow_html=True)
    if any(DEVICE_TYPE_SCOPE.get(d) == "whole_system" for d in order):
        st.caption("(*) pomiar `whole_system` - caly SoC / komputer, nie sam uklad. Serie szrafowane na wykresach.")


def figure_block(
    fig: go.Figure,
    *,
    name_parts: list[str],
    key: str,
    co_pokazuje: str,
    jak_czytac: str,
    co_wynika: str,
    hipoteza: str | None = None,
    caption: str | None = None,
) -> None:
    """Wykres + przyciski eksportu (PNG 300 DPI / SVG) + blok interpretacyjny. Jeden
    wywolanie na kazdy wykres w dashboardzie."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if caption:
        st.caption(caption)
    render_figure_export_buttons(fig, [p for p in name_parts if p], key_prefix=key)
    interpretation(co_pokazuje, jak_czytac, co_wynika, hipoteza)


def table_block(
    df: pd.DataFrame,
    *,
    name_parts: list[str],
    key: str,
    caption: str | None = None,
    height: int = 380,
    rename_pl: bool = True,
) -> None:
    """Tabela w kontenerze z przewijaniem + eksport CSV/Excel (opisowa nazwa pliku)."""
    if df is None or df.empty:
        st.info("Brak wierszy do pokazania dla wybranych filtrow.")
        return
    shown = df.rename(columns={c: label(c) for c in df.columns}) if rename_pl else df
    st.dataframe(shown, width="stretch", height=height, hide_index=True)
    if caption:
        st.caption(caption)
    render_dataframe_export_buttons(df, [p for p in name_parts if p], key_prefix=key)
