"""Agregacja i analiza po stronie dashboardu - rzetelnie, z jawnym `n`.

Zasady (z wymagan pracy):
- energie NA PROBKE agregujemy dwustopniowo: najpierw srednia po `repetition_group_id`
  (niezalezne powtorzenia tej samej kombinacji), potem srednia/odch.std. po tych srednich
  w obrebie grupy wykresu (device_type x task x phase x ...),
- slupki bledow = odchylenie standardowe (ddof=1, jak w runnerze: statistics.stdev),
- przy malym `n` (< 3) oznaczamy wynik jako niepewny (`small_n`),
- nie usredniamy nic bez zwrocenia `n`.

`run_stats_summary` (liczone przez runner) jest zrodlem dla energii CALKOWITEJ; tu liczymy
z surowych `runs`, bo potrzebujemy tez energii NA PROBKE i dowolnych przekrojow."""

from __future__ import annotations

import pandas as pd

SMALL_N_THRESHOLD = 3  # n < 3 -> odchylenie std. praktycznie nieinformacyjne

_META_COLUMNS = [
    "task",
    "model_name",
    "phase",
    "precision",
    "batch_size",
    "device_device_label",
    "device_short",
    "device_device_type",
    "device_type_label",
    "device_category",
    "arch_class",
    "device_machine_name",
    "measurement_scope",
    "scope_label",
    "is_whole_system",
    "device_row_id",
]


def aggregate_by_repetition_group(
    df: pd.DataFrame, value_col: str, mean_col: str, std_col: str
) -> pd.DataFrame:
    """Jeden wiersz na `repetition_group_id`: mean/std dla `value_col`, `n_repetitions`,
    plus kolumny opisowe z pierwszego wiersza grupy. Std z pojedynczego powtorzenia -> 0.0
    (ujednolicone z runnerem)."""
    if "repetition_group_id" not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    usable = df[df[value_col].notna() & df["repetition_group_id"].notna()].copy()
    if usable.empty:
        return pd.DataFrame()

    meta_cols = [c for c in _META_COLUMNS if c in usable.columns]
    grouped = usable.groupby("repetition_group_id")[value_col].agg(["mean", "std", "count"])
    grouped = grouped.rename(columns={"mean": mean_col, "std": std_col, "count": "n_repetitions"})
    grouped[std_col] = grouped[std_col].fillna(0.0)
    meta = usable.groupby("repetition_group_id")[meta_cols].first()
    return grouped.join(meta).reset_index()


def group_means(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Srednia `value_col` po kazdym `repetition_group_id` (stopien 1 agregacji), z
    kolumnami opisowymi. Kolumna wynikowa: `group_mean`, plus `n_repetitions`."""
    return aggregate_by_repetition_group(df, value_col, "group_mean", "group_std")


def agg_metric(
    df: pd.DataFrame,
    value_col: str,
    by: list[str],
    per_group_first: bool = True,
) -> pd.DataFrame:
    """Agregacja metryki `value_col` w przekroju kolumn `by`.

    per_group_first=True (domyslnie, dla metryk energii): najpierw srednia po
    `repetition_group_id`, potem statystyka po tych srednich. Zwraca kolumny:
    `by...`, `mean`, `std`, `sem`, `n` (liczba grup powtorzen albo przebiegow), `n_runs`,
    `small_n` (bool, n < 3). `std`/`sem` z jednego elementu -> 0.0."""
    by = [c for c in by if c in df.columns]
    if not by or value_col not in df.columns:
        return pd.DataFrame()

    base = df[df[value_col].notna()].copy()
    if base.empty:
        return pd.DataFrame()

    if per_group_first and "repetition_group_id" in base.columns:
        stage1 = (
            base.groupby(by + ["repetition_group_id"], observed=True, dropna=False)[value_col]
            .mean()
            .reset_index()
        )
        runs_per_cell = (
            base.groupby(by, observed=True, dropna=False)[value_col].size().rename("n_runs")
        )
    else:
        stage1 = base[by + [value_col]].copy()
        runs_per_cell = (
            base.groupby(by, observed=True, dropna=False)[value_col].size().rename("n_runs")
        )

    out = (
        stage1.groupby(by, observed=True, dropna=False)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    out["sem"] = (out["std"] / out["n"].pow(0.5)).fillna(0.0)
    out = out.merge(runs_per_cell.reset_index(), on=by, how="left")
    out["small_n"] = out["n"] < SMALL_N_THRESHOLD
    return out


def agg_two_metrics(
    df: pd.DataFrame, metric_a: str, metric_b: str, by: list[str], per_group_first: bool = True
) -> pd.DataFrame:
    """Srednie DWOCH metryk w tym samym przekroju `by` - do wykresu Pareto (przepustowosc
    vs energia/probke). Zwraca `by...`, `metric_a`, `metric_b`, `n`, `n_runs`."""
    by = [c for c in by if c in df.columns]
    if not by or metric_a not in df.columns or metric_b not in df.columns:
        return pd.DataFrame()
    a = agg_metric(df, metric_a, by=by, per_group_first=per_group_first)
    b = agg_metric(df, metric_b, by=by, per_group_first=per_group_first)
    if a.empty or b.empty:
        return pd.DataFrame()
    a = a.rename(columns={"mean": metric_a})[by + [metric_a, "n", "n_runs", "small_n"]]
    b = b.rename(columns={"mean": metric_b})[by + [metric_b]]
    return a.merge(b, on=by, how="inner")


def pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_better: str = "higher",
    y_better: str = "lower",
) -> pd.Series:
    """Zwraca boolean Series (index jak `df`): True = punkt na froncie Pareto (nie jest
    zdominowany przez zaden inny). Domyslnie: wieksze X lepsze (przepustowosc), mniejsze Y
    lepsze (J/probke)."""
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=bool)

    xs = sub[x_col].to_numpy()
    ys = sub[y_col].to_numpy()
    x_sign = 1.0 if x_better == "higher" else -1.0
    y_sign = 1.0 if y_better == "higher" else -1.0
    x_adj = xs * x_sign
    y_adj = ys * y_sign

    on_front = []
    for i in range(len(sub)):
        dominated = (
            (x_adj >= x_adj[i]) & (y_adj >= y_adj[i]) & ((x_adj > x_adj[i]) | (y_adj > y_adj[i]))
        ).any()
        on_front.append(not dominated)
    return pd.Series(on_front, index=sub.index).reindex(df.index, fill_value=False)


def build_decision_table(
    df: pd.DataFrame,
    value_col: str = "energy_per_sample_j",
    group_col: str = "device_type_label",
    device_types: list[str] | None = None,
) -> pd.DataFrame:
    """Dla kazdego scenariusza (task, model, phase, batch_size): zwyciezca po sredniej
    `value_col` (mniej = lepiej) wsrod wartosci `group_col` (domyslnie `device_type_label`,
    mozna dac `arch_class`), przewaga nad kolejnym (%), n grup powtorzen, czy w grze byly
    serie `whole_system`.

    `device_types` - jesli podane, ogranicza porownanie do tych `device_device_type`
    (np. `["cpu","cuda","rocm"]` - tylko pomiar `device_only`, uczciwszy).
    `llm_inference` wykluczone (num_iterations skrajnie zmienne)."""
    needed = {"task", "model_name", "phase", "batch_size", group_col, value_col, "device_device_type"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    base = df[df["task"].ne("llm_inference") & df[value_col].notna()].copy()
    if device_types is not None:
        base = base[base["device_device_type"].isin(device_types)]
    if base.empty:
        return pd.DataFrame()

    cells = agg_metric(base, value_col, by=["task", "model_name", "phase", "batch_size", group_col])
    if cells.empty:
        return pd.DataFrame()

    ws_flag = (
        base.groupby(["task", "model_name", "phase", "batch_size"], observed=True)["is_whole_system"]
        .any().rename("ma_whole_system")
        if "is_whole_system" in base.columns else None
    )

    rows = []
    scenario_cols = ["task", "model_name", "phase", "batch_size"]
    for scenario, grp in cells.groupby(scenario_cols, observed=True):
        grp = grp.sort_values("mean")
        winner = grp.iloc[0]
        runner_up = grp.iloc[1] if len(grp) > 1 else None
        margin_pct = (
            (runner_up["mean"] - winner["mean"]) / runner_up["mean"] * 100.0
            if runner_up is not None and runner_up["mean"] else None
        )
        row = dict(zip(scenario_cols, scenario))
        row.update({
            "zwyciezca": winner[group_col],
            "zwyciezca_j_na_probke": winner["mean"],
            "kolejny": runner_up[group_col] if runner_up is not None else None,
            "kolejny_j_na_probke": runner_up["mean"] if runner_up is not None else None,
            "przewaga_pct": margin_pct,
            "n_porownywanych": len(grp),
            "n_grup_zwyciezcy": int(winner["n"]),
            "small_n": bool(winner["small_n"]),
            "porownywane": ", ".join(grp[group_col].astype(str)),
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    if ws_flag is not None and not result.empty:
        result = result.merge(ws_flag.reset_index(), on=scenario_cols, how="left")
    return result.sort_values(scenario_cols).reset_index(drop=True)


def same_machine_metric(
    df: pd.DataFrame, value_col: str = "energy_per_sample_j"
) -> pd.DataFrame:
    """Najuczciwsze porownanie: w obrebie JEDNEJ maszyny zestawia CPU z akceleratorami tej
    samej maszyny w scenariuszach (task, model, phase, batch_size, precision), ktore
    wykonaly OBA. Zwraca wiersze z `mean`/`std`/`n` metryki `value_col` per
    (machine, scenario, device_type) + `ratio_vs_cpu` (mean / mean_cpu; <1 = akcelerator
    oszczedniejszy, >1 = drozszy)."""
    keys = ["device_machine_name", "task", "model_name", "phase", "batch_size", "precision"]
    if not set(keys + ["device_device_type", value_col]).issubset(df.columns):
        return pd.DataFrame()

    base = df[df[value_col].notna()].copy()
    by = keys + ["device_device_type", "device_type_label", "arch_class", "is_whole_system"]
    cells = agg_metric(base, value_col, by=[c for c in by if c in base.columns])
    if cells.empty:
        return pd.DataFrame()

    # zostaw tylko scenariusze, w ktorych na tej maszynie byl CPU i co najmniej 1 akcelerator
    def _keep(g: pd.DataFrame) -> bool:
        types = set(g["device_device_type"])
        return "cpu" in types and len(types) > 1

    scen_keys = ["device_machine_name", "task", "model_name", "phase", "batch_size", "precision"]
    cells = cells.groupby(scen_keys, observed=True).filter(_keep)
    if cells.empty:
        return pd.DataFrame()

    cpu_mean = (
        cells[cells["device_device_type"] == "cpu"].set_index(scen_keys)["mean"].rename("cpu_mean")
    )
    cells = cells.merge(cpu_mean, left_on=scen_keys, right_index=True, how="left")
    cells["ratio_vs_cpu"] = cells["mean"] / cells["cpu_mean"]
    _task_abbr = {"image_classification": "img", "nlp_sentiment": "nlp", "llm_inference": "llm"}
    cells["scenario"] = (
        cells["task"].map(lambda t: _task_abbr.get(t, t)) + " · " + cells["model_name"].astype(str)
        + " · " + cells["phase"].str[:3] + " · b" + cells["batch_size"].astype("Int64").astype(str)
        + " · " + cells["precision"].astype(str)
    )
    return cells.sort_values(scen_keys + ["mean"])


def same_machine_winners(
    df: pd.DataFrame, value_col: str = "energy_per_sample_j"
) -> pd.DataFrame:
    """Zwyciezca energetyczny W OBREBIE JEDNEJ MASZYNY dla kazdego scenariusza
    (machine, task, model, phase, batch_size, precision), sposrod urzadzen tej maszyny,
    ktore ten scenariusz wykonaly (>=2 urzadzenia). To jest uczciwe porownanie - ten sam
    krzem, ten sam pomiar. Zwraca 1 wiersz na scenariusz: `zwyciezca` (device_type_label),
    `kolejny`, `przewaga_pct`, `zwyciezca_device_type`, `cpu_wygral` (bool), `ma_whole_system`,
    `n_grup_zwyciezcy`, `small_n`. `llm_inference` wykluczone."""
    keys = ["device_machine_name", "task", "model_name", "phase", "batch_size", "precision"]
    if not set(keys + ["device_device_type", value_col]).issubset(df.columns):
        return pd.DataFrame()

    base = df[df["task"].ne("llm_inference") & df[value_col].notna()].copy()
    by = keys + ["device_device_type", "device_type_label", "arch_class", "is_whole_system"]
    cells = agg_metric(base, value_col, by=[c for c in by if c in base.columns])
    if cells.empty:
        return pd.DataFrame()
    cells = cells.groupby(keys, observed=True).filter(lambda g: g["device_device_type"].nunique() >= 2)
    if cells.empty:
        return pd.DataFrame()

    rows = []
    for scenario, grp in cells.groupby(keys, observed=True):
        grp = grp.sort_values("mean")
        w, r = grp.iloc[0], (grp.iloc[1] if len(grp) > 1 else None)
        rows.append({
            **dict(zip(keys, scenario)),
            "zwyciezca": w["device_type_label"],
            "zwyciezca_device_type": w["device_device_type"],
            "kolejny": r["device_type_label"] if r is not None else None,
            "przewaga_pct": ((r["mean"] - w["mean"]) / r["mean"] * 100.0) if (r is not None and r["mean"]) else None,
            "zwyciezca_j_na_probke": w["mean"],
            "kolejny_j_na_probke": r["mean"] if r is not None else None,
            "cpu_wygral": w["device_device_type"] == "cpu",
            "ma_whole_system": bool(grp["is_whole_system"].any()) if "is_whole_system" in grp else False,
            "n_grup_zwyciezcy": int(w["n"]),
            "small_n": bool(w["small_n"]),
            "porownywane": ", ".join(grp["device_type_label"].astype(str)),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def crossover_batches(
    line_df: pd.DataFrame, x_col: str, y_col: str, series_col: str, reference: str = "cpu"
) -> list[dict]:
    """Wykrywa, przy jakim `x_col` (batch size) krzywa `reference` (domyslnie CPU) przecina
    krzywa innej serii - czyli gdzie zmienia sie, ktora architektura jest oszczedniejsza.
    Zwraca liste {seria, x_lewy, x_prawy, x_przeciecia} (interpolacja liniowa)."""
    if not {x_col, y_col, series_col}.issubset(line_df.columns):
        return []
    ref = line_df[line_df[series_col] == reference].sort_values(x_col)
    if ref.empty:
        return []
    ref_map = dict(zip(ref[x_col], ref[y_col]))

    crossings = []
    for series, grp in line_df[line_df[series_col] != reference].groupby(series_col, observed=True):
        grp = grp.sort_values(x_col)
        xs = [x for x in grp[x_col] if x in ref_map]
        for left, right in zip(xs, xs[1:]):
            d_left = ref_map[left] - grp.loc[grp[x_col] == left, y_col].iloc[0]
            d_right = ref_map[right] - grp.loc[grp[x_col] == right, y_col].iloc[0]
            if d_left == 0 or (d_left < 0) != (d_right < 0):
                frac = 0.0 if (d_right - d_left) == 0 else d_left / (d_left - d_right)
                crossings.append(
                    {
                        "seria": series,
                        "x_lewy": left,
                        "x_prawy": right,
                        "x_przeciecia": left + frac * (right - left),
                    }
                )
    return crossings


# ---------------------------------------------------------------------------
# Dynamiczne listy z DataFrame - zeby nowe maszyny / device_type pojawialy sie same
# ---------------------------------------------------------------------------
def list_machines(df: pd.DataFrame) -> list[str]:
    col = "device_machine_name"
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []


def list_device_types(df: pd.DataFrame, order: list[str] | None = None) -> list[str]:
    col = "device_device_type"
    if col not in df.columns:
        return []
    present = set(df[col].dropna().unique())
    if order:
        ordered = [d for d in order if d in present]
        return ordered + sorted(present - set(ordered))
    return sorted(present)


def list_tasks(df: pd.DataFrame, order: list[str] | None = None) -> list[str]:
    if "task" not in df.columns:
        return []
    present = set(df["task"].dropna().unique())
    if order:
        return [t for t in order if t in present] + sorted(present - set(order))
    return sorted(present)


def coverage_matrix(df: pd.DataFrame, index: str, columns: list[str], value: str = "id") -> pd.DataFrame:
    """Liczba przebiegow w przekroju `index` x (polaczone `columns`) - do heatmapy pokrycia
    na stronie 0 (od razu widac luki i maszyny, ktore jeszcze nie doslaly wynikow)."""
    cols_present = [c for c in columns if c in df.columns]
    if index not in df.columns or not cols_present or value not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_col"] = tmp[cols_present].astype(str).agg(" / ".join, axis=1)
    return tmp.pivot_table(index=index, columns="_col", values=value, aggfunc="count", fill_value=0)
