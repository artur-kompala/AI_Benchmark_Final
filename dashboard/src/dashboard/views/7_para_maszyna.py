"""Strona 7 - CPU vs akcelerator na TEJ SAMEJ maszynie. Najuczciwsze porownanie: ten sam
krzem, ten sam workload, ta sama chwila - tylko inny uklad wykonujacy obliczenia."""

from __future__ import annotations

import streamlit as st

from dashboard.components.charts import paired_bars_same_machine
from dashboard.components.format import fmt_ratio
from dashboard.components.page import figure_block, growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.data_access.aggregation import list_machines, same_machine_metric
from dashboard.i18n.pl import label

INTRO = """
Porownania miedzy maszynami zawsze mieszaja wiele zmiennych (inny CPU, inny zasilacz,
inny pomiar). Tutaj patrzymy **w obrebie jednej maszyny**: dla scenariuszy, ktore na tej
maszynie wykonaly zarowno CPU, jak i akcelerator, zestawiamy energie na probke i liczymy,
**ile razy** akcelerator jest oszczedniejszy (albo drozszy) od CPU tej samej maszyny.
"""

header("CPU vs akcelerator na tej samej maszynie", INTRO)
df = load_runs_or_stop()
df = df[df["task"] != "llm_inference"]

pairs = same_machine_metric(df, "energy_per_sample_j")
if pairs.empty:
    st.warning("Brak maszyn ze scenariuszami wykonanymi jednoczesnie na CPU i akceleratorze.")
    st.stop()

machines = sorted(pairs["device_machine_name"].dropna().unique())
machine = st.selectbox("Maszyna", machines)
render_device_legend(sorted(df[df["device_machine_name"] == machine]["device_device_type"].dropna().unique()))

m = pairs[pairs["device_machine_name"] == machine].copy()
devs = ", ".join(sorted(m["device_type_label"].unique()))

fig = paired_bars_same_machine(
    m, value_col="mean", error_col="std", x_col="scenario",
    y_title=label("energy_per_sample_j"), log_y=True,
    title=f"Energia na probke — {machine} ({devs})",
)

# podsumowanie krotnosci akcelerator vs CPU
acc = m[m["device_device_type"] != "cpu"]
ratio_lines = []
for dtl, g in acc.groupby("device_type_label", observed=True):
    r = g["ratio_vs_cpu"].dropna()
    if not r.empty:
        med = float(r.median())
        verdict = "oszczedniej" if med < 1 else "drozej"
        ratio_lines.append(f"**{dtl}**: mediana {fmt_ratio(1 / med if med < 1 else med)} {verdict} od CPU "
                           f"(zakres {fmt_ratio(1 / r.max() if r.max() < 1 else r.max())})")
ratio_txt = "; ".join(ratio_lines) if ratio_lines else "brak par do porownania"

figure_block(
    fig,
    name_parts=["para_maszyna", machine],
    key="pair",
    co_pokazuje=(
        f"Srednia energia na probke dla kazdego scenariusza wykonanego na maszynie **{machine}** "
        f"zarowno przez CPU, jak i akcelerator(y). Slupki zgrupowane po scenariuszu, kolor = architektura. "
        f"Skala Y logarytmiczna. Podsumowanie krotnosci: {ratio_txt}."
    ),
    jak_czytac=(
        "W kazdej grupie porownaj slupek CPU z akceleratorem. Slupek akceleratora nizszy = akcelerator "
        "oszczedniejszy w tym scenariuszu. `b1`/`b4` w nazwie = male batche. Slupki szrafowane (iGPU/NPU) "
        "to `whole_system`: ich energia jest zawyzona zakresem pomiaru, wiec realna przewaga CPU jest "
        "tam WIEKSZA niz widac."
    ),
    co_wynika=(
        "Wynik silnie zalezy od tego, JAKI to akcelerator. **Laptopowe GPU NVIDIA (CUDA)** wygrywaja z "
        "CPU we wszystkich scenariuszach (przewaga topnieje przy batch=1, ale zostaje). **Desktopowe GPU "
        "AMD (ROCm, RX 7800 XT)** przegrywa z CPU tej samej maszyny w wielu scenariuszach fp16 i dla "
        "DistilBERT - wysoki pobor bazowy karty nie oplaca sie przy lekkim zadaniu. **iGPU/NPU** - po "
        "korekcie o `whole_system` roznica z CPU jest mala."
    ),
    hipoteza=(
        "Bezposredni test hipotezy na najczystszych danych. Potwierdza sie **waska wersja**: CPU bije "
        "akcelerator, gdy akcelerator jest przewymiarowany do zadania (desktopowe GPU, lekki model, maly "
        "batch). Wobec sprzetu dobranego do warunkow ograniczonych zasobow (GPU laptopowe, NPU) CPU "
        "pozostaje mniej efektywny."
    ),
)

st.subheader("Tabela porownan parami")
cols = ["scenario", "device_type_label", "arch_class", "mean", "std", "n", "ratio_vs_cpu", "is_whole_system"]
tbl = m[[c for c in cols if c in m.columns]].rename(columns={"mean": "energy_per_sample_j"})
table_block(tbl, name_parts=["para_maszyna_tabela", machine], key="pair_tbl",
            caption="ratio_vs_cpu < 1 = akcelerator oszczedniejszy; > 1 = drozszy od CPU tej samej maszyny.")
growing_note()
