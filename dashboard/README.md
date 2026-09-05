# Dashboard analityczny (Streamlit)

Aplikacja Streamlit do analizy wynikow benchmarku z pracy magisterskiej
*„Analiza energooszczednosci i wydajnosci nowoczesnych ukladow obliczeniowych
w zastosowaniach sztucznej inteligencji"*.

Czyta dane **wylacznie z Supabase na zywo** przez `supabase-py`. Dashboard **nigdy** nie
zapisuje do bazy ani nie uruchamia benchmarkow - to zadanie `benchmark_runner` (osobny,
niezalezny proces; patrz [`../docs/architecture.md`](../docs/architecture.md)).
**Nie ma trybu offline, importu z plikow ani przelacznika zrodla** - zbior danych wciaz
przyrasta (trwaja testy na kolejnych maszynach), wiec dashboard zawsze pokazuje aktualny
stan z Supabase. Listy maszyn, urzadzen i kombinacji sa wyliczane z danych przy kazdym
odswiezeniu (cache `ttl=60 s`), wiec **nowe maszyny i przebiegi pojawiaja sie bez zmian
w kodzie** - m.in. maszyny w drodze: *wojtek* (CPU + GPU AMD), *milosz* (CPU + GPU NVIDIA),
*maciej* (CPU + NPU + iGPU).

## Struktura (11 zakladek, `src/dashboard/views/`)

Router: `src/dashboard/app.py` (`st.navigation`). Kolejnosc jak w menu bocznym:

| # | Zakladka | Co pokazuje |
|---|---|---|
| 0 | **Start / Jak czytac** | Hipoteza pracy, legenda kolorow `device_type`, wyjasnienie `device_only` vs `whole_system`, KPI (przebiegi, maszyny, urzadzenia, zadania, zakres dat), heatmapa pokrycia `typ urzadzenia x zadanie x faza`. |
| 1 | **Efektywnosc energetyczna** | `J/probke` i `probki/dzul` wg architektury, panele per zadanie, slupki + slupki bledow, przelacznik `precision`/`phase`. Rdzen hipotezy. |
| 2 | **Wydajnosc** | `throughput [probki/s]` i `czas [s]` wg architektury - kontrast do str. 1 (surowa szybkosc GPU). |
| 3 | **Kompromis Pareto** | scatter `przepustowosc` (X) vs `J/probke` (Y, log-log), kolor+ksztalt = `device_type`, panel per zadanie, zaznaczony front Pareto. |
| 4 | **Wplyw batch size** | `J/probke` i `throughput` vs `batch_size` (linie per `device_type`), panel per model; zaznaczone przeciecia krzywej CPU. |
| 5 | **Wplyw precyzji** | `J/probke` i `accuracy` obok siebie per architektura; `int8` = tylko CPU, `intel_gpu_openvino`/`npu_openvino` = tylko `fp32`; status kwantyzacji INT8. Mini-sekcja `llm_inference` (eksploracyjnie). |
| 6 | **Moc chwilowa i temperatura** | rozklady `avg/peak_power_watts` (box) per architektura, `peak` vs `avg`, `avg_temperature_c` gdzie zmierzona. |
| 7 | **CPU vs akcelerator (ta sama maszyna)** | najuczciwsze porownanie parami w obrebie jednej maszyny; slupki `J/probke` + krotnosc „ile razy oszczedniej/drozej vs CPU". Pary budowane dynamicznie. |
| 8 | **Co mierzymy / powtarzalnosc** | pomiar programowy vs watomierz Tuya, `power_discrepancy_pct` z komentarzem „roznica zakresu, nie blad"; nacisk na `coefficient_of_variation_energy`; liczba probek mocy vs prog; `measurement_scope`; anomalia `flops_per_watt` ROCm. |
| 9 | **Weryfikacja hipotezy / wnioski** | zwyciezca energetyczny **w obrebie maszyny** (`same_machine_winners`), rozklad per maszyna, tabela porownan + jawne „co potwierdza / co oslabia hipoteze". |
| 10 | **Eksport danych** | pelna tabela `runs`+`devices`+`run_summary` z filtrami, `run_stats_summary`, tabele decyzyjne - CSV/Excel. |

Pod **kazdym** wykresem jest blok **Co pokazuje / Jak czytac / Co z tego wynika**
(z odniesieniem do hipotezy) - dashboard prowadzi do wniosku, nie wysypuje surowych wykresow.

## System wizualny

- **Jedna mapa kolorow `device_type`** (`cpu`, `cuda`, `rocm`, `amd_igpu`,
  `intel_gpu_openvino`, `npu_openvino`) - paleta z skilla `dataviz` (walidowana pod
  daltonizm), ta sama na kazdym wykresie (`i18n/pl.py::DEVICE_TYPE_COLORS`).
- **Architektura** (4 klasy: CPU / GPU dyskretne / GPU zintegrowane (iGPU) /
  NPU (akcelerator dedykowany)) - wymiar grupujacy w hipotezie. NPU to **trzecia
  architektura** obok CPU i GPU (praca par. 1.2), nie „GPU".
- Serie `whole_system` (`amd_igpu`, `intel_gpu_openvino`, `npu_openvino`) sa **szrafowane**
  i opisane „caly SoC / komputer" - nie zestawia sie ich wprost z `device_only`
  (`cpu`, `cuda`, `rocm`) bez zastrzezenia.
- Kazda os z jednostka; skala **logarytmiczna** tam, gdzie rozrzut jest rzedu wielkosci.
- Slupki bledow z odchylenia std. (agregacja dwustopniowa: mean po `repetition_group_id`,
  potem po konfiguracjach); przy `n < 3` slupek przygaszony, `n` zawsze w tooltipie.
- Liczby formatowane po polsku (`components/format.py`).

## Eksport (nie ruszany mechanizm, tylko rozszerzany)

Kazdy wykres: **PNG >= 300 DPI** (gotowe do druku w pracy) lub **SVG**.
Kazda tabela: **CSV / Excel**. Opisowa nazwa pliku (np.
`efektywnosc_j_na_probke_fp32_inference_2026-09-04.png`) - `components/export.py`.

## Instalacja

```bash
python -m venv venv
venv/bin/pip install -r requirements/requirements-dashboard.txt
venv/bin/pip install -e .
```

Skopiuj `.env.example` do `.env` i uzupelnij `SUPABASE_URL` / `SUPABASE_KEY`
(te same co w `benchmark_runner/.env` - patrz [`../docs/setup_supabase.md`](../docs/setup_supabase.md)).
Dashboard tylko czyta; do czasu wdrozenia RLS mozna uzyc tego samego klucza co runner.

## Uruchomienie

```bash
venv/bin/streamlit run src/dashboard/app.py
```

Domyslnie `http://localhost:8501`. Motyw jasny jest wymuszony w `.streamlit/config.toml`
(deterministyczne kolory - wykresy trafiaja wprost do pracy).

## Testy

```bash
venv/bin/python -m pytest
```

Offline (bez sieci), na atrapie klienta Supabase (`tests/conftest.py`). Pokrywaja m.in.:
klasyfikacje `amd_igpu` -> „GPU zintegrowane (iGPU)" i `npu_openvino` ->
„NPU (akcelerator dedykowany)", kolumny pochodne (`enrich`), agregacje po grupach
powtorzen, front Pareto, tabele decyzyjne, porownania w obrebie maszyny, dynamiczne
listy maszyn/urzadzen, stronicowanie zapytan (> 1000 wierszy), formatowanie PL.

## Stan danych (chwilowy, rosnie)

~1260 przebiegow, 6 maszyn, 6 typow urzadzen, 3 zadania
(`image_classification`, `nlp_sentiment`, `llm_inference`), zakres `2026-08` .. `2026-09`.
`llm_inference` jest **poza glownymi porownaniami** (`num_iterations` skrajnie zmienne,
brak realnych grup powtorzen, tylko CPU) - trafia do osobnej sekcji eksploracyjnej na str. 5.
