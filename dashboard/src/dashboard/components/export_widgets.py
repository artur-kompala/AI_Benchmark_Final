"""Widgety eksportu (przyciski pobierania) wspoldzielone przez wszystkie strony -
kazdy wykres i kazda tabela w dashboardzie ma mozliwosc eksportu (PNG >=300 DPI dla
wykresow, CSV/Excel dla danych zrodlowych), z opisowa nazwa pliku (patrz
components/export.py:build_export_filename)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.export import (
    build_export_filename,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    figure_to_png_bytes,
    figure_to_svg_bytes,
)


def render_figure_export_buttons(fig: go.Figure, name_parts: list[str], key_prefix: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Pobierz wykres jako PNG (300 DPI)",
            data=figure_to_png_bytes(fig),
            file_name=build_export_filename(*name_parts, ext="png"),
            mime="image/png",
            key=f"{key_prefix}_png",
        )
    with col2:
        st.download_button(
            "Pobierz wykres jako SVG (wektorowy)",
            data=figure_to_svg_bytes(fig),
            file_name=build_export_filename(*name_parts, ext="svg"),
            mime="image/svg+xml",
            key=f"{key_prefix}_svg",
        )


def render_dataframe_export_buttons(df: pd.DataFrame, name_parts: list[str], key_prefix: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Pobierz dane jako CSV",
            data=dataframe_to_csv_bytes(df),
            file_name=build_export_filename(*name_parts, ext="csv"),
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    with col2:
        st.download_button(
            "Pobierz dane jako Excel",
            data=dataframe_to_excel_bytes(df),
            file_name=build_export_filename(*name_parts, ext="xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx",
        )
