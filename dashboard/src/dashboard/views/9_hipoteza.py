"""Strona 9 - Weryfikacja hipotezy / wnioski. Zwyciezca energetyczny liczymy W OBREBIE
JEDNEJ MASZYNY (uczciwe porownanie: ten sam krzem, ten sam pomiar), a potem zbieramy
wyniki. Plus jawne "co potwierdza / co oslabia hipoteze"."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.export_widgets import render_dataframe_export_buttons, render_figure_export_buttons
from dashboard.components.format import fmt_pct_pl, fmt_pl
from dashboard.components.narrative import interpretation
from dashboard.components.page import growing_note, header, load_runs_or_stop, render_device_legend, table_block
from dashboard.components.theme import apply_house_style, device_color_map
from dashboard.data_access.aggregation import same_machine_winners
from dashboard.i18n.pl import DEVICE_TYPE_LABELS, DEVICE_TYPE_ORDER, TASK_LABELS, label, phase_label

INTRO = """
Zwyciezce energetycznego liczymy **w obrebie jednej maszyny** - dla scenariuszy, ktore na
danej maszynie wykonaly co najmniej dwa rozne urzadzenia. To jedyne uczciwe porownanie:
ten sam procesor bazowy, ten sam zasilacz, ten sam sposob pomiaru. `llm_inference`
wykluczone. Serie `whole_system` (iGPU, NPU) oznaczone - maja inny zakres pomiaru.
"""

header("Weryfikacja hipotezy / wnioski", INTRO, show_hypothesis=True)
df = load_runs_or_stop()
render_device_legend(sorted(df["device_device_type"].dropna().unique()))

w = same_machine_winners(df, "energy_per_sample_j")
if w.empty:
    st.warning("Za malo danych do porownan w obrebie maszyn.")
    st.stop()

present_types = [t for t in DEVICE_TYPE_ORDER if t in df["device_device_type"].unique()]
color_map = {DEVICE_TYPE_LABELS.get(t, t): c for t, c in device_color_map(present_types).items()}
label_order = [DEVICE_TYPE_LABELS.get(t, t) for t in present_types]

total = len(w)
cpu_wins = int(w["cpu_wygral"].sum())
cpu_machines = sorted(w.loc[w["cpu_wygral"], "device_machine_name"].unique())

# --- 1. kto wygrywa ile ---
st.subheader("Kto wygrywa ile porownan (w obrebie maszyny)")
wins = w["zwyciezca"].value_counts().reindex(label_order).dropna().reset_index()
wins.columns = ["typ", "liczba"]
wins_order = [t for t in label_order if t in set(wins["typ"])]
fig_wins = px.bar(wins, x="liczba", y="typ", orientation="h", color="typ",
                  color_discrete_map=color_map, category_orders={"typ": wins_order[::-1]},
                  title=f"Zwyciezcy energetyczni — {total} porownan w obrebie maszyn")
fig_wins.update_traces(marker_line_width=0, showlegend=False)
apply_house_style(fig_wins, title=fig_wins.layout.title.text, x_title="liczba scenariuszy", y_title="", show_legend=False)
st.plotly_chart(fig_wins, width="stretch", config={"displayModeBar": False})
render_figure_export_buttons(fig_wins, ["hipoteza_zwyciezcy"], key_prefix="wins")
interpretation(
    co_pokazuje=f"Ile z {total} porownan (scenariusz x maszyna, min. 2 urzadzenia) wygrywa energetycznie "
    "kazdy typ urzadzenia - najnizsza srednia energia na probke.",
    jak_czytac="Dluzszy slupek = ten typ czesciej najoszczedniejszy w bezposrednim starciu na tej samej maszynie.",
    co_wynika=(
        f"**CPU wygrywa {cpu_wins} z {total}** porownan - i wszystkie na maszynie "
        f"{', '.join(cpu_machines)} (Ryzen 7 5700X3D + desktopowa Radeon RX 7800 XT). Na maszynach z "
        f"laptopowym GPU NVIDIA (CUDA) oraz z iGPU CPU nie wygrywa ani razu."
    ),
    hipoteza="Hipoteza w formie slabej ('sa scenariusze, w ktorych CPU bije GPU') - **potwierdzona**, "
    "ale waska: dotyczy przewymiarowanego desktopowego GPU przy lekkich zadaniach. W formie mocnej "
    "(CPU dorownuje najlepszemu dostepnemu GPU) - **nie potwierdzona**.",
)

# --- 2. rozklad per maszyna ---
st.subheader("Zwyciezcy wg maszyny")
per_m = w.groupby(["device_machine_name", "zwyciezca"], observed=True).size().reset_index(name="liczba")
fig_m = px.bar(per_m, x="liczba", y="device_machine_name", color="zwyciezca", orientation="h",
               color_discrete_map=color_map, category_orders={"zwyciezca": label_order},
               title="Ile scenariuszy wygrywa ktore urzadzenie, maszyna po maszynie")
fig_m.update_traces(marker_line_width=0)
apply_house_style(fig_m, title=fig_m.layout.title.text, x_title="liczba scenariuszy", y_title="",
                  legend_title=label("device_type"))
st.plotly_chart(fig_m, width="stretch", config={"displayModeBar": False})
render_figure_export_buttons(fig_m, ["hipoteza_zwyciezcy_per_maszyna"], key_prefix="perm")
interpretation(
    co_pokazuje="Ten sam wynik rozbity na maszyny - slupek skladany pokazuje, ktore urzadzenie danej "
    "maszyny wygrywalo scenariusze.",
    jak_czytac="Maszyny z GPU NVIDIA (ola, proksza) - caly slupek to CUDA. artur-PC (GPU AMD desktop) - "
    "duzy udzial CPU. artur-MS-7B86 (iGPU + NPU Intel) - iGPU/NPU.",
    co_wynika="Efekt 'CPU bije GPU' jest zjawiskiem **jednej maszyny** - tej z najbardziej "
    "przewymiarowanym GPU wzgledem zadania. To nie jest wlasciwosc CPU jako klasy, tylko relacji "
    "CPU <-> konkretny, zbyt mocny akcelerator.",
    hipoteza="Praca dotyczy 'warunkow ograniczonych zasobow'. RX 7800 XT to sprzet raczej NIE z tej "
    "kategorii - i wlasnie tam CPU sie oplaca. To niuansuje hipoteze: liczy sie dobor GPU do zadania.",
)

# --- 3. szczegoly wygranych CPU ---
if cpu_wins:
    st.subheader(f"Scenariusze, w ktorych CPU wygral ({cpu_wins})")
    cpu_tbl = w[w["cpu_wygral"]].copy()
    cpu_tbl["przewaga nad kolejnym"] = cpu_tbl["przewaga_pct"].map(lambda v: fmt_pct_pl(v, 1))
    cpu_tbl["CPU [J/probke]"] = cpu_tbl["zwyciezca_j_na_probke"].map(lambda v: fmt_pl(v, 4))
    cpu_tbl["kolejny [J/probke]"] = cpu_tbl["kolejny_j_na_probke"].map(lambda v: fmt_pl(v, 4))
    view = cpu_tbl.rename(columns={"device_machine_name": "maszyna", "task": "zadanie", "model_name": "model",
                                   "phase": "faza", "batch_size": "batch", "precision": "precyzja",
                                   "kolejny": "przegral (kolejny)", "n_grup_zwyciezcy": "n"})
    st.dataframe(view[["maszyna", "zadanie", "model", "faza", "batch", "precyzja", "przegral (kolejny)",
                       "przewaga nad kolejnym", "CPU [J/probke]", "kolejny [J/probke]", "n"]],
                 width="stretch", height=300, hide_index=True)
    render_dataframe_export_buttons(cpu_tbl, ["hipoteza_wygrane_cpu"], key_prefix="cpuwin")
    st.caption("Przewaga do ~50% dla `fp16` na desktopowym GPU AMD - tam CPU jest realnie oszczedniejszy. "
               "Dla `fp32` przewagi bywaja drobne (0,4-10%) - to raczej parytet niz wyrazna wygrana CPU.")

# --- 4. pelna tabela ---
st.subheader("Pelna tabela porownan w obrebie maszyn")
full = w.copy()
full["przewaga"] = full["przewaga_pct"].map(lambda v: fmt_pct_pl(v, 1))
full = full.rename(columns={"device_machine_name": "maszyna", "task": "zadanie", "model_name": "model",
                            "phase": "faza", "batch_size": "batch", "precision": "precyzja",
                            "n_grup_zwyciezcy": "n", "ma_whole_system": "whole_system w grze", "small_n": "n<3"})
st.dataframe(full[["maszyna", "zadanie", "model", "faza", "batch", "precyzja", "zwyciezca", "przewaga",
                   "kolejny", "n", "n<3", "whole_system w grze", "porownywane"]],
             width="stretch", height=420, hide_index=True)
render_dataframe_export_buttons(w, ["tabela_decyzyjna_ta_sama_maszyna"], key_prefix="dt")

# --- 5. wnioski jawne ---
st.divider()
st.subheader("Co potwierdza, co oslabia hipoteze")
st.markdown(
    f"""
**Potwierdza hipoteze (w waskiej, warunkowej formie):**
- **CPU vs desktopowe GPU AMD (ROCm, RX 7800 XT), ta sama maszyna:** CPU wygrywa **{cpu_wins} z 44**
  porownan na tej maszynie - dla `fp16` z przewaga do ~50%, dla DistilBERT `fp32` 9-42%, a nawet w
  czesci scenariuszy treningu. Przewymiarowana karta desktopowa ma wysoki pobor bazowy i przy lekkim
  zadaniu nie oplaca sie energetycznie - to jest teza pracy o "warunkach ograniczonych zasobow".
- **Strona 4 (batch size):** krzywa energii CPU przecina krzywa ROCm - ponizej pewnego batcha CPU jest nizej.
- **iGPU / NPU po korekcie o zakres pomiaru:** ich energia to caly SoC, wiec realna roznica CPU vs iGPU
  jest mniejsza niz surowe slupki (str. 1, 8).

**Oslabia / ogranicza hipoteze:**
- **CPU vs laptopowe GPU NVIDIA (CUDA):** CPU nie wygrywa **ani razu** (0 z ~68 porownan). Roznica od
  ~1,6x (mobilenet, batch 1) do 30-50x (resnet50 / trening).
- **Strona 3 (Pareto):** zaden punkt CPU nie lezy na froncie Pareto w zadnym zadaniu.
- **Strony 1-2:** usredniajac, GPU dyskretne i NPU sa 1-2 rzedy wielkosci oszczedniejsze i szybsze.
- Efekt "CPU bije GPU" wystepuje **tylko na jednej maszynie** - to relacja CPU <-> konkretny zbyt mocny
  akcelerator, nie wlasciwosc CPU jako klasy.

**Za malo danych, by przesadzic:**
- **NPU:** 1 maszyna, tylko klasyfikacja obrazow, `whole_system`, n=2 na grupe - wygrywa 24 porownania
  mimo handicapu zakresu, ale to wynik wstepny.
- **ROCm:** 1 maszyna (desktop), anomalia FLOPS/W (str. 8) - kierunek wiarygodny, skala do weryfikacji.
  Maszyna *wojtek* (drugie AMD GPU) potwierdzi lub obali te obserwacje.
- **iGPU Intel:** tylko `fp32` + klasyfikacja obrazow.
- Maszyny *wojtek*, *milosz*, *maciej* jeszcze nie doslaly wynikow.

**Wniosek koncowy:** hipoteza w mocnej formie (CPU >= najlepszy dostepny GPU) **nie jest potwierdzona**.
W formie slabszej ("istnieja realne scenariusze, w ktorych CPU jest energetycznie oszczedniejszy od GPU")
**jest potwierdzona**, ale waska: dotyczy przewymiarowanego desktopowego GPU przy lekkich zadaniach,
niskiej precyzji i malych/srednich batchach. Wobec sprzetu faktycznie dobranego do warunkow
ograniczonych zasobow (energooszczedne GPU laptopowe, NPU) CPU pozostaje mniej efektywny energetycznie.
"""
)
growing_note()
