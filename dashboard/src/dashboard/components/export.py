"""Eksport wykresow (PNG w rozdzielczosci >=300 DPI, przez kaleido) i tabel (CSV/Excel)
do pobrania z dashboardu - gotowe do wklejenia bezposrednio do pracy dyplomowej."""

from __future__ import annotations

import re
from datetime import date
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go

# plotly/kaleido traktuje width/height jako "CSS px" w odniesieniu 96 px/cal, a `scale`
# jako mnoznik do faktycznej liczby pikseli w pliku - stad DPI = 96 * scale. Docelowe
# 300 DPI przy planowanym rozmiarze wydruku 10x6 cali (typowy wykres w pracy dyplomowej,
# szerszy niz wysoki) daje wyjsciowy PNG 3000x1800 px.
_TARGET_DPI = 300
_CSS_DPI = 96
_EXPORT_WIDTH_IN = 10
_EXPORT_HEIGHT_IN = 6
_EXPORT_WIDTH = round(_EXPORT_WIDTH_IN * _CSS_DPI)
_EXPORT_HEIGHT = round(_EXPORT_HEIGHT_IN * _CSS_DPI)
_EXPORT_SCALE = _TARGET_DPI / _CSS_DPI

_FILENAME_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9_\-]+")


def build_export_filename(*parts: str, ext: str) -> str:
    """Opisowa nazwa pliku eksportu (np. 'ranking_energii_image_classification_2026-08-15.png')
    zamiast generycznej 'chart.png' - laczy podane czesci (np. nazwa wykresu, zadanie,
    urzadzenie) i biezaca date, sanityzujac znaki niebezpieczne dla systemu plikow."""
    safe_parts = [_FILENAME_UNSAFE_CHARS.sub("_", str(p)).strip("_") for p in parts if p]
    safe_parts.append(date.today().isoformat())
    return "_".join(safe_parts) + f".{ext}"


def figure_to_png_bytes(fig: go.Figure) -> bytes:
    """PNG >=300 DPI (patrz stale powyzej) - wystarczajace do wklejenia bezposrednio do
    dokumentu pracy bez utraty jakosci przy druku."""
    return fig.to_image(format="png", width=_EXPORT_WIDTH, height=_EXPORT_HEIGHT, scale=_EXPORT_SCALE)


def figure_to_svg_bytes(fig: go.Figure) -> bytes:
    return fig.to_image(format="svg", width=_EXPORT_WIDTH, height=_EXPORT_HEIGHT)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "wyniki") -> bytes:
    # Excel/openpyxl nie obsluguje datetimes ze strefa czasowa (nasze kolumny
    # started_at/finished_at/created_at/computed_at sa w UTC) - usuwamy tzinfo
    # (wartosci pozostaja poprawne, tylko bez jawnej strefy w komorce Excela).
    df = df.copy()
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_localize(None)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()
