"""Jednolity system wizualny wykresow (Plotly) - reguly z skilla `dataviz` przeniesione
na Plotly/Streamlit:

- jedna paleta `device_type` (walidowana pod daltonizm), stala kolejnosc slotow,
- osie zawsze z jednostka, siatka = cienka, ciagla linia (bez kreskowania),
- legenda pozioma nad wykresem, wieksza czcionka, oddech w marginesach,
- serie `whole_system` (iGPU / NPU) szrafowane + osobny dopisek,
- slupki/punkty z n < 3 - mniejsze krycie,
- zero chartjunk: bez 3D, bez kol, bez podwojnej osi Y.

Kazdy builder w components/charts.py konczy wywolaniem `apply_house_style`."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.i18n.pl import (
    DEVICE_TYPE_COLORS,
    DEVICE_TYPE_LABELS,
    DEVICE_TYPE_ORDER,
    DEVICE_TYPE_SYMBOLS,
    DEVICE_TYPE_UNKNOWN_COLOR,
)

# --- kolory chrome (jasny motyw - deterministyczny, bo wykresy ida wprost do pracy) ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#ffffff"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT_STACK = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

SMALL_N_OPACITY = 0.4
FULL_OPACITY = 0.92
WHOLE_SYSTEM_PATTERN = "/"  # szrafura 45 stopni dla serii whole_system

# status / akcenty (nie mieszac z paleta serii)
ACCENT_GOOD = "#0ca30c"
ACCENT_WARN = "#fab219"
ACCENT_CRITICAL = "#d03b3b"
REFLINE = "#898781"


# kolory 4 architektur (tabela decyzyjna / mapa zwyciezcow) - spojne z paleta device_type:
# CPU jak cpu, GPU dyskretne jak cuda, iGPU jak amd_igpu, NPU jak npu_openvino.
ARCH_CLASS_ORDER_COLORS = {
    "CPU": "#2a78d6",
    "GPU dyskretne": "#eb6834",
    "GPU zintegrowane (iGPU)": "#eda100",
    "NPU (akcelerator dedykowany)": "#008300",
    "GPU (kategoria nieznana)": "#898781",
}


def device_color(device_type: str) -> str:
    return DEVICE_TYPE_COLORS.get(device_type, DEVICE_TYPE_UNKNOWN_COLOR)


def device_symbol(device_type: str) -> str:
    return DEVICE_TYPE_SYMBOLS.get(device_type, "circle")


def device_color_map(device_types: list[str] | None = None) -> dict[str, str]:
    keys = device_types or DEVICE_TYPE_ORDER
    return {d: device_color(d) for d in keys}


def ordered_device_types(present: list[str]) -> list[str]:
    """Zachowuje stala kolejnosc slotow palety, pomija to, czego nie ma w danych."""
    present_set = set(present)
    return [d for d in DEVICE_TYPE_ORDER if d in present_set] + sorted(present_set - set(DEVICE_TYPE_ORDER))


def apply_house_style(
    fig: go.Figure,
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    log_y: bool = False,
    log_x: bool = False,
    legend_title: str | None = None,
    height: int | None = None,
    show_legend: bool = True,
) -> go.Figure:
    # faceting -> wiele osi X; legenda 'h' na gorze zderza sie z paskami tytulow paneli,
    # wiec przy facetach wrzucamy legende na dol.
    is_faceted = sum(1 for k in fig.layout if k.startswith("xaxis")) > 1
    legend_bottom = is_faceted and show_legend

    top_margin = 96 if title else 44
    if show_legend and not legend_bottom:
        top_margin += 30
    bottom_margin = 108 if legend_bottom else 64

    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_STACK, size=14, color=INK_PRIMARY),
        title=dict(text=title, font=dict(size=17, color=INK_PRIMARY), x=0.0, xanchor="left",
                   y=0.98, yanchor="top", yref="container"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=76, r=28, t=top_margin, b=bottom_margin),
        bargap=0.28,
        bargroupgap=0.12,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(family=FONT_STACK, size=13)),
        legend=dict(
            orientation="h",
            yanchor="top" if legend_bottom else "bottom",
            y=-0.22 if legend_bottom else 1.02,
            yref="paper",
            xanchor="left",
            x=0.0,
            title=dict(text=legend_title or "", font=dict(size=12, color=INK_SECONDARY)),
            font=dict(size=12, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=show_legend,
    )
    if height:
        fig.update_layout(height=height)

    axis_common = dict(
        gridcolor=GRID,
        griddash="solid",
        gridwidth=1,
        zerolinecolor=AXIS,
        zerolinewidth=1,
        linecolor=AXIS,
        ticks="outside",
        tickcolor=AXIS,
        ticklen=5,
        title_font=dict(size=13, color=INK_SECONDARY),
        tickfont=dict(size=12, color=INK_SECONDARY),
        automargin=True,
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)
    # na osi logarytmicznej "zeroline" celuje w y=0 => log(0) = -inf i rozwala autorange;
    # wylaczamy ja jawnie (siatka i tak niesie odczyt).
    if log_y:
        fig.update_yaxes(type="log", zeroline=False)
    if log_x:
        fig.update_xaxes(type="log", zeroline=False)

    # tytuly osi: przy facetach px powiela je pod kazdym panelem - zbijamy do jednego,
    # wysrodkowanego pod calym rzedem paneli (mniej szumu, zgodnie z dataviz).
    if is_faceted:
        if x_title is not None:
            fig.update_xaxes(title_text="")
            if x_title:
                fig.add_annotation(text=x_title, showarrow=False, xref="paper", yref="paper",
                                   x=0.5, y=-0.16 if not legend_bottom else -0.30, xanchor="center",
                                   font=dict(size=13, color=INK_SECONDARY))
        if y_title is not None:
            fig.update_yaxes(title_text="")
            if y_title:
                fig.add_annotation(text=y_title, showarrow=False, xref="paper", yref="paper",
                                   x=-0.055, y=0.5, textangle=-90, yanchor="middle",
                                   font=dict(size=13, color=INK_SECONDARY))
    else:
        if x_title is not None:
            fig.update_xaxes(title_text=x_title)
        if y_title is not None:
            fig.update_yaxes(title_text=y_title)

    # facety: skracamy 'kolumna=wartosc' do samej wartosci; tekst juz przetlumaczony
    # przez charts._facet_labels tam, gdzie trzeba - tu tylko styl czcionki + fallback.
    def _ann(a):
        if "=" in a.text and a.text.split("=", 1)[0].isidentifier():
            a.update(text=a.text.split("=", 1)[-1])
        a.update(font=dict(size=13, color=INK_SECONDARY))
    fig.for_each_annotation(_ann)
    fig.update_layout(margin_pad=4)
    return fig


def mark_whole_system_traces(fig: go.Figure, whole_system_types: set[str]) -> go.Figure:
    """Szrafuje slupki serii `whole_system` (klucz po nazwie serii = etykieta device_type
    albo surowy device_type). Reszta traces bez zmian."""
    ws_labels = {DEVICE_TYPE_LABELS.get(t, t) for t in whole_system_types} | set(whole_system_types)
    for tr in fig.data:
        if getattr(tr, "name", None) in ws_labels and tr.type == "bar":
            tr.update(marker_pattern_shape=WHOLE_SYSTEM_PATTERN, marker_pattern_solidity=0.12)
    return fig


def add_identity_line(fig: go.Figure, lo: float, hi: float, *, text: str | None = None) -> go.Figure:
    """Linia y = x jako TRACE (nie shape) - dziala poprawnie na osiach logarytmicznych
    (shape na log-osi wymaga wspolrzednych log10 i psuje zakres)."""
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=REFLINE, width=1, dash="dot"),
        hoverinfo="skip", showlegend=False, name="y = x",
    ))
    if text:
        fig.add_annotation(x=hi, y=lo, text=text, showarrow=False, xanchor="right", yanchor="bottom",
                           font=dict(size=10, color=INK_MUTED))
    return fig


def add_reference_line(fig: go.Figure, *, y=None, x=None, text: str | None = None, color: str = REFLINE):
    if y is not None:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=text, annotation_position="top left",
                      annotation_font=dict(size=11, color=INK_MUTED))
    if x is not None:
        fig.add_vline(x=x, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=text, annotation_position="top",
                      annotation_font=dict(size=11, color=INK_MUTED))
    return fig
