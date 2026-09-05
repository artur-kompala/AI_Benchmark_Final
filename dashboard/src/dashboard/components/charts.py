"""Buildery wykresow (Plotly) dla dashboardu - male, proste figury, wspolny system
wizualny (components/theme.py), etykiety po polsku (i18n/pl.py).

Konwencje:
- kazda funkcja zwraca `go.Figure`; przy braku danych zwraca czytelny placeholder
  (`empty_figure`) zamiast wyjatku (spojne z reszta projektu),
- kolor serii ZAWSZE po `device_type` (jedna mapa z theme.py),
- serie `whole_system` (iGPU/NPU) szrafowane,
- os Y z jednostka; skala log tam, gdzie rozrzut jest rzedu wielkosci,
- tooltip pokazuje wartosc + n.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.theme import (
    AXIS,
    GRID,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    apply_house_style,
    device_color_map,
    device_symbol,
    mark_whole_system_traces,
    ordered_device_types,
)
from dashboard.i18n.pl import (
    DEVICE_TYPE_LABELS,
    DEVICE_TYPE_ORDER,
    DEVICE_TYPE_SCOPE,
    label,
    phase_label,
    task_label,
)

_WHOLE_SYSTEM_TYPES = {t for t, s in DEVICE_TYPE_SCOPE.items() if s == "whole_system"}


def empty_figure(message: str = "Brak danych dla wybranych filtrow") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=INK_MUTED),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=240,
        xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _facet_labels(fig: go.Figure) -> None:
    """px pisze etykiety paneli jako 'kolumna=wartosc' - tniemy prefiks i tlumaczymy
    znane wartosci (task / phase) na polski."""
    def _relabel(a):
        raw = a.text.split("=", 1)[-1].strip()
        if raw in {"image_classification", "nlp_sentiment", "llm_inference"}:
            a.update(text=task_label(raw))
        elif raw in {"train", "inference"}:
            a.update(text=phase_label(raw))
        else:
            a.update(text=raw)
    fig.for_each_annotation(_relabel)


def _device_type_labels_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "device_type_label" not in df.columns and "device_device_type" in df.columns:
        df["device_type_label"] = df["device_device_type"].map(lambda t: DEVICE_TYPE_LABELS.get(t, t))
    return df


# ---------------------------------------------------------------------------
# 1. Slupki poziome: metryka wg device_type (error bary, panele, szrafura whole_system)
#    Poziomo - dlugie nazwy architektur nie wymagaja obracania etykiet.
# ---------------------------------------------------------------------------
def bar_metric_by_device(
    agg_df: pd.DataFrame,
    *,
    value_col: str = "mean",
    error_col: str | None = "std",
    n_col: str = "n",
    small_n_col: str = "small_n",
    facet_col: str | None = None,
    facet_row: str | None = None,
    y_title: str,
    title: str | None = None,
    log_y: bool = False,
    category_order: list[str] | None = None,
) -> go.Figure:
    if agg_df is None or agg_df.empty or value_col not in agg_df.columns:
        return empty_figure()

    data = _device_type_labels_col(agg_df)
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}
    n_facets = data[facet_col].nunique() if facet_col and facet_col in data.columns else 1

    hover_cols = [c for c in [n_col, "n_runs", "scope_label"] if c in data.columns]
    # skala log wymaga dodatniej dolnej granicy wasa - zabezpieczenie przed <=0
    err = error_col if (error_col and error_col in data.columns) else None

    fig = px.bar(
        data,
        y="device_type_label",
        x=value_col,
        color="device_type_label",
        orientation="h",
        error_x=err,
        facet_col=facet_col,
        facet_row=facet_row,
        category_orders={
            "device_type_label": label_order[::-1],  # pierwszy slot palety na gorze
            **({facet_col: category_order} if facet_col and category_order else {}),
        },
        color_discrete_map=color_map_by_label,
        custom_data=hover_cols,
        title=title,
    )

    tip = [f"<b>%{{x:.4g}}</b> {y_title}"]
    for i, c in enumerate(hover_cols):
        tip.append(f"{label(c)}: %{{customdata[{i}]}}")
    fig.update_traces(hovertemplate="<br>".join(tip) + "<extra>%{y}</extra>",
                      marker_line_width=0, error_x=dict(color=AXIS, thickness=1.2, width=4))

    # maly n (< 3): przygaszony slupek - identyfikacja i tak jest z osi Y, nie z koloru
    if small_n_col in data.columns:
        small_labels = set(data.loc[data[small_n_col].fillna(False), "device_type_label"])
        for tr in fig.data:
            if getattr(tr, "name", None) in small_labels and set(data[data["device_type_label"] == tr.name][small_n_col].fillna(False)) == {True}:
                tr.update(opacity=0.45)

    mark_whole_system_traces(fig, _WHOLE_SYSTEM_TYPES)
    _facet_labels(fig)
    apply_house_style(fig, title=title, x_title=y_title, y_title="", log_x=log_y,
                      legend_title=label("device_type"), show_legend=False,
                      height=170 + 44 * len(type_order) + (36 if n_facets > 1 else 0))
    return fig


# ---------------------------------------------------------------------------
# 2. Linie: metryka vs batch_size, linia per device_type, panel per model/task
# ---------------------------------------------------------------------------
def line_metric_vs_batch(
    agg_df: pd.DataFrame,
    *,
    value_col: str = "mean",
    error_col: str | None = "std",
    facet_col: str | None = "model_name",
    y_title: str,
    title: str | None = None,
    log_y: bool = True,
    crossings: list[dict] | None = None,
) -> go.Figure:
    if agg_df is None or agg_df.empty or value_col not in agg_df.columns:
        return empty_figure()

    data = _device_type_labels_col(agg_df).sort_values("batch_size")
    data["batch_size_str"] = data["batch_size"].astype("Int64").astype(str)
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}

    fig = px.line(
        data,
        x="batch_size_str",
        y=value_col,
        color="device_type_label",
        facet_col=facet_col,
        markers=True,
        error_y=error_col if (error_col and error_col in data.columns) else None,
        category_orders={"device_type_label": label_order,
                         "batch_size_str": [str(b) for b in sorted(data["batch_size"].dropna().unique())]},
        color_discrete_map=color_map_by_label,
        custom_data=[c for c in ["n", "n_runs"] if c in data.columns],
        title=title,
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=9, line=dict(color=SURFACE, width=1.5)),
                      error_y=dict(thickness=1, width=0, color=AXIS))
    # szrafura nie dotyczy linii - dla whole_system dajemy linie przerywana
    ws_labels = {DEVICE_TYPE_LABELS.get(t, t) for t in _WHOLE_SYSTEM_TYPES}
    for tr in fig.data:
        if tr.name in ws_labels:
            tr.update(line=dict(width=2, dash="dot"))

    if crossings:
        for c in crossings:
            fig.add_vline(x=c.get("x_przeciecia"), line=dict(color=INK_MUTED, width=1, dash="dot"))

    _facet_labels(fig)
    apply_house_style(fig, title=title, x_title=label("batch_size"), y_title=y_title, log_y=log_y,
                      legend_title=label("device_type"))
    return fig


# ---------------------------------------------------------------------------
# 3. Pareto: przepustowosc (X) vs energia/probke (Y, log), kolor+symbol = device_type
# ---------------------------------------------------------------------------
def pareto_scatter(
    points_df: pd.DataFrame,
    *,
    x_col: str = "throughput_samples_per_s",
    y_col: str = "energy_per_sample_j",
    facet_col: str | None = "task",
    front_col: str = "on_front",
    title: str | None = None,
) -> go.Figure:
    if points_df is None or points_df.empty or x_col not in points_df.columns:
        return empty_figure()

    data = _device_type_labels_col(points_df)
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}
    symbol_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): device_symbol(t) for t in type_order}

    hover = [c for c in ["device_short", "model_name", "phase", "batch_size", "precision", "n_runs", "scope_label"]
             if c in data.columns]
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="device_type_label",
        symbol="device_type_label",
        facet_col=facet_col,
        category_orders={"device_type_label": label_order},
        color_discrete_map=color_map_by_label,
        symbol_map=symbol_map_by_label,
        custom_data=hover,
        title=title,
        log_y=True,
        log_x=True,
    )
    fig.update_traces(marker=dict(size=11, line=dict(color=SURFACE, width=1.5)), opacity=0.9)
    tip = [f"<b>%{{y:.4g}}</b> {label(y_col)}", f"%{{x:.4g}} {label(x_col)}"]
    for i, c in enumerate(hover):
        tip.append(f"{label(c)}: %{{customdata[{i}]}}")
    fig.update_traces(hovertemplate="<br>".join(tip) + "<extra>%{fullData.name}</extra>")

    # front Pareto - schodkowa linia laczaca punkty niezdominowane, per panel
    if front_col in data.columns:
        facets = list(dict.fromkeys(data[facet_col])) if facet_col else [None]
        for idx, fv in enumerate(facets):
            sub = data[data[front_col]]
            if fv is not None:
                sub = sub[sub[facet_col] == fv]
            sub = sub.sort_values(x_col)
            if len(sub) >= 2:
                axis_suffix = "" if idx == 0 else str(idx + 1)
                fig.add_trace(go.Scatter(
                    x=sub[x_col], y=sub[y_col], mode="lines", line=dict(color=INK_SECONDARY, width=1.5, dash="dash"),
                    name="front Pareto", showlegend=(idx == 0), hoverinfo="skip",
                    xaxis=f"x{axis_suffix}", yaxis=f"y{axis_suffix}",
                ))

    _facet_labels(fig)
    apply_house_style(fig, title=title, x_title=f"{label(x_col)}  (wiecej = szybciej)",
                      y_title=f"{label(y_col)}  (nizej = oszczedniej)", legend_title=label("device_type"))
    return fig


# ---------------------------------------------------------------------------
# 4. Rozklad mocy: box per device_type
# ---------------------------------------------------------------------------
def box_metric_by_device(
    df: pd.DataFrame,
    *,
    y_col: str,
    y_title: str,
    title: str | None = None,
    facet_col: str | None = None,
    log_y: bool = False,
) -> go.Figure:
    if df is None or df.empty or y_col not in df.columns or df[y_col].notna().sum() == 0:
        return empty_figure()

    data = _device_type_labels_col(df[df[y_col].notna()])
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}

    fig = px.box(
        data,
        y="device_type_label",
        x=y_col,
        color="device_type_label",
        orientation="h",
        facet_col=facet_col,
        category_orders={"device_type_label": label_order[::-1]},
        color_discrete_map=color_map_by_label,
        points="outliers",
        title=title,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.45), line=dict(width=1.5), boxmean=True)
    # go.Box nie wspiera szrafury (fillpattern) - rozroznienie whole_system niosa legenda
    # kolorow nad wykresem (z gwiazdka) + podpis strony.
    _facet_labels(fig)
    apply_house_style(fig, title=title, x_title=y_title, y_title="", log_x=log_y,
                      legend_title=label("device_type"), show_legend=False,
                      height=150 + 44 * len(type_order) + (30 if facet_col else 0))
    return fig


# ---------------------------------------------------------------------------
# 5. Heatmapa pokrycia (strona 0)
# ---------------------------------------------------------------------------
def coverage_heatmap(matrix_df: pd.DataFrame, *, title: str | None = None,
                     x_title: str = "", y_title: str = "") -> go.Figure:
    if matrix_df is None or matrix_df.empty:
        return empty_figure("Brak danych do heatmapy pokrycia")

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        color_continuous_scale=["#f4f7fb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"],
        aspect="auto",
        title=title,
    )
    fig.update_traces(textfont=dict(size=12), xgap=2, ygap=2,
                      hovertemplate="%{y} · %{x}<br><b>%{z}</b> przebiegow<extra></extra>")
    fig.update_coloraxes(showscale=False)
    apply_house_style(fig, title=title, x_title=x_title, y_title=y_title, show_legend=False)
    fig.update_xaxes(tickangle=-25, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# 6. Pary "ta sama maszyna": slupki metryki wg scenariusza, kolor = device_type
# ---------------------------------------------------------------------------
def paired_bars_same_machine(
    agg_df: pd.DataFrame,
    *,
    value_col: str = "mean",
    error_col: str | None = "std",
    x_col: str = "scenario",
    y_title: str,
    title: str | None = None,
    log_y: bool = True,
) -> go.Figure:
    if agg_df is None or agg_df.empty or value_col not in agg_df.columns:
        return empty_figure()

    data = _device_type_labels_col(agg_df)
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}

    n_scen = data[x_col].nunique()
    fig = px.bar(
        data,
        y=x_col,
        x=value_col,
        color="device_type_label",
        orientation="h",
        barmode="group",
        error_x=error_col if (error_col and error_col in data.columns) else None,
        category_orders={"device_type_label": label_order},
        color_discrete_map=color_map_by_label,
        custom_data=[c for c in ["n", "n_runs", "ratio_vs_cpu"] if c in data.columns],
        title=title,
    )
    fig.update_traces(marker_line_width=0, error_x=dict(color=AXIS, thickness=1.2, width=4))
    mark_whole_system_traces(fig, _WHOLE_SYSTEM_TYPES)
    apply_house_style(fig, title=title, x_title=y_title, y_title="", log_x=log_y,
                      legend_title=label("device_type"),
                      height=160 + 34 * n_scen * max(data["device_type_label"].nunique(), 1) // 2)
    return fig


# ---------------------------------------------------------------------------
# 7. Scatter ogolny (strona 8: rozbieznosc w czasie, probki mocy)
# ---------------------------------------------------------------------------
def scatter_over_time(
    df: pd.DataFrame,
    *,
    y_col: str,
    y_title: str,
    title: str | None = None,
    ref_y: float | None = None,
    ref_text: str | None = None,
) -> go.Figure:
    if df is None or df.empty or y_col not in df.columns or df[y_col].notna().sum() == 0:
        return empty_figure()

    data = _device_type_labels_col(df[df[y_col].notna()])
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}
    symbol_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): device_symbol(t) for t in type_order}

    fig = px.scatter(
        data,
        x="started_at",
        y=y_col,
        color="device_type_label",
        symbol="device_type_label",
        category_orders={"device_type_label": label_order},
        color_discrete_map=color_map_by_label,
        symbol_map=symbol_map_by_label,
        custom_data=[c for c in ["device_short", "task", "model_name", "phase", "precision", "batch_size"] if c in data.columns],
        title=title,
    )
    fig.update_traces(marker=dict(size=9, line=dict(color=SURFACE, width=1)), opacity=0.85)
    if ref_y is not None:
        fig.add_hline(y=ref_y, line=dict(color=INK_MUTED, width=1, dash="dot"),
                      annotation_text=ref_text, annotation_position="top left",
                      annotation_font=dict(size=11, color=INK_MUTED))
    apply_house_style(fig, title=title, x_title=label("started_at"), y_title=y_title,
                      legend_title=label("device_type"))
    return fig


# ---------------------------------------------------------------------------
# 8. Slupki poziome: pojedyncza metryka wg device_type (np. CV energii)
# ---------------------------------------------------------------------------
def hbar_metric_by_device(
    agg_df: pd.DataFrame,
    *,
    value_col: str,
    x_title: str,
    title: str | None = None,
    error_col: str | None = None,
    n_col: str = "n",
) -> go.Figure:
    if agg_df is None or agg_df.empty or value_col not in agg_df.columns:
        return empty_figure()

    data = _device_type_labels_col(agg_df).sort_values(value_col)
    present_types = data["device_device_type"].dropna().unique().tolist()
    type_order = ordered_device_types(present_types)
    label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in type_order]
    color_map_by_label = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(type_order).items()}

    fig = px.bar(
        data,
        x=value_col,
        y="device_type_label",
        color="device_type_label",
        orientation="h",
        error_x=error_col if (error_col and error_col in data.columns) else None,
        category_orders={"device_type_label": label_order[::-1]},
        color_discrete_map=color_map_by_label,
        custom_data=[c for c in [n_col, "n_runs"] if c in data.columns],
        title=title,
    )
    fig.update_traces(marker_line_width=0)
    mark_whole_system_traces(fig, _WHOLE_SYSTEM_TYPES)
    apply_house_style(fig, title=title, x_title=x_title, y_title="", legend_title=label("device_type"),
                      show_legend=False)
    return fig


# ---------------------------------------------------------------------------
# 9. Grupowane slupki: dwie metryki obok siebie (np. energia vs accuracy dla int8)
# ---------------------------------------------------------------------------
def grouped_bar_generic(
    agg_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    y_title: str,
    title: str | None = None,
    color_map: dict | None = None,
    barmode: str = "group",
    error_col: str | None = None,
    facet_col: str | None = None,
    category_orders: dict | None = None,
    log_y: bool = False,
    y_range: list | None = None,
    orientation: str = "v",
) -> go.Figure:
    if agg_df is None or agg_df.empty or y_col not in agg_df.columns:
        return empty_figure()

    horizontal = orientation == "h"
    err = error_col if (error_col and error_col in agg_df.columns) else None
    fig = px.bar(
        agg_df,
        x=y_col if horizontal else x_col,
        y=x_col if horizontal else y_col,
        color=color_col,
        orientation=orientation,
        barmode=barmode,
        facet_col=facet_col,
        error_x=err if horizontal else None,
        error_y=None if horizontal else err,
        color_discrete_map=color_map or {},
        category_orders=category_orders or {},
        custom_data=[c for c in ["n", "n_runs"] if c in agg_df.columns],
        title=title,
    )
    fig.update_traces(marker_line_width=0)
    fig.update_traces(error_x=dict(color=AXIS, thickness=1.2, width=4)) if horizontal else \
        fig.update_traces(error_y=dict(color=AXIS, thickness=1.2, width=4))
    _facet_labels(fig)
    apply_house_style(
        fig, title=title,
        x_title=y_title if horizontal else label(x_col),
        y_title="" if horizontal else y_title,
        log_x=log_y and horizontal, log_y=log_y and not horizontal,
        legend_title=label(color_col),
        height=(150 + 42 * agg_df[x_col].nunique()) if horizontal else None,
    )
    if y_range:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(range=y_range)
    if not horizontal:
        fig.update_xaxes(tickangle=-20)
    return fig
