# IMPLEMENTATION_STATUS.md — przebudowa dashboardu (praca magisterska)

> Źródło prawdy o postępie. Czego tu nie ma — uznajemy za niezrobione.
> Dopisywać przyrostowo, datować wpisy, odwoływać się do plików ścieżką.
> Kolejna sesja: przeczytaj **Stan ogólny → W trakcie → TODO** i kontynuuj.

---

## Stan ogólny (2026-09-05)

Przebudowa **zakończona i zweryfikowana w nowej sesji** (kod z 2026-09-04 był już
kompletny, ale nieodpalony na świeżo — ta sesja uruchomiła aplikację na żywo,
przeszła wszystkie 11 stron w przeglądarce z realnymi danymi z Supabase, przepuściła
paletę kolorów przez walidator skilla `dataviz` i naprawiła 3 realne, potwierdzone
usterki (patrz „Zrobione" niżej). Wszystkie 47 testów zielone. Router `st.navigation`
w `app.py`, strony w `src/dashboard/views/`. Zmiany z obu sesji są **nadal
niescommitowane** (working tree) — `git log -- dashboard/` pokazuje tylko commit
sprzed przebudowy.

### KLUCZOWY WYNIK ANALIZY (do narracji i podsumowania końcowego)
Analiza „w obrębie maszyny" (`same_machine_winners`, 136 porównań scenariusz×maszyna,
≥2 urządzenia) — to jest uczciwa podstawa, NIE pooled decision table (pooled zawsze
wygrywa CUDA, bo uśrednia 2 energooszczędne laptopy).

- Zwycięzcy: **CUDA 68, ROCm 25, iGPU Intel 24, CPU 19** (z 136).
- **CPU wygrywa 19/136 — WSZYSTKIE na `artur-PC`** (Ryzen 7 5700X3D + desktop RX 7800 XT),
  gdzie CPU bije GPU w 19 z 44 scenariuszy tej maszyny (43%). `fp16`: przewaga CPU do ~50%.
  DistilBERT `fp32`: 9–42%. Nawet część treningu.
- Na maszynach z **laptopowym GPU NVIDIA (CUDA)** i z **iGPU** CPU nie wygrywa **ani razu**.
- Wniosek: efekt „CPU bije GPU" to zjawisko **jednej maszyny** — relacja CPU ↔ konkretny
  PRZEWYMIAROWANY desktopowy GPU przy lekkim zadaniu. Nie własność CPU jako klasy.
- Hipoteza w formie MOCNEJ (CPU ≥ najlepszy dostępny GPU) — **NIE potwierdzona**
  (0 punktów CPU na froncie Pareto; vs CUDA CPU zawsze 1,6–50× gorszy).
- Hipoteza w formie SŁABEJ („są realne scenariusze, gdzie CPU jest oszczędniejszy od GPU")
  — **potwierdzona, ale wąska**: przewymiarowany desktopowy GPU + lekkie zadanie + fp16
  + mały/średni batch. To pasuje do tezy pracy o „warunkach ograniczonych zasobów"
  (RX 7800 XT NIE jest sprzętem z tej kategorii).
- NPU: 24 wygrane mimo handicapu `whole_system` → mocny kandydat na efektywność
  lekkiego inference, ale 1 maszyna / tylko obrazy / n=2.

Sekcja „co potwierdza / osłabia" na str. 9 już to odzwierciedla. NIE naciągać.

Hipoteza pracy: efektywność energetyczna zależy od architektury (CPU/GPU/NPU) i parametrów;
**dla modeli o umiarkowanej złożoności i małych batchach CPU może dorównać lub przewyższyć
GPU w energii/próbkę, mimo niższej wydajności.** Metryka rozstrzygająca: **J/próbkę**.

Rozpoznanie zakończone (kod + skill `dataviz` + dane w Supabase). Zaczynam warstwę danych
i komponenty.

## Zrobione

- [x] Przeczytana praca magisterska — hipoteza + terminologia.
- [x] Przeczytany cały obecny kod `dashboard/**` i testy; wczytany skill `dataviz`.
- [x] Rozpoznane dane w Supabase (projekt `fvkelwyullnratnfkesv`) — patrz
  **Charakterystyka danych** niżej.
- [x] `components/format.py` — formatowanie liczb po polsku (fmt_pl, fmt_sci_pl, fmt_ratio,
  fmt_pct_pl, fmt_signed_pct_pl, fmt_range_pl).
- [x] `i18n/pl.py` — przepisane: `device_category_label` naprawione (amd_igpu→iGPU,
  npu→„NPU (akcelerator dedykowany)"), `DEVICE_TYPE_*` (paleta dataviz sloty 1–6,
  symbole), `ARCH_CLASS_*`, `SCOPE_*` + `SCOPE_EXPLAINER`, `PRECISION_*` (rampa),
  `TASK_LABELS`/`PHASE_LABELS`, NAV 0–10, `COLUMN_LABELS` z jednostkami, teksty narracji.
- [x] `data_access/enrich.py` (NOWY) — `enrich_runs_df`: `arch_class`, `device_type_label`,
  `energy_per_sample_j`, `energy_per_1000_samples_j`, `samples_per_joule`, `is_whole_system`,
  `scope_label`, `device_short` (czyszczenie nazw CPU/GPU + ujednoznacznienie po maszynie),
  smart_plug 0.0 → NaN (+ zerowanie power_discrepancy_pct gdy brak realnego odczytu).
- [x] `data_access/queries.py` — `get_runs_df`/`get_run_stats_summary_df` stronicowane
  (`_fetch_all` + `.range()`, obejście limitu 1000 PostgREST — WAŻNE, wcześniej ucinało
  do 1000), wołają `enrich_runs_df`.
- [x] `data_access/aggregation.py` — `agg_metric` (dwustopniowy, mean/std/sem/n/small_n),
  `agg_two_metrics`, `pareto_front`, `build_decision_table` (+ filtr arch_classes),
  `same_machine_metric`, `crossover_batches`, `list_machines/list_device_types/list_tasks`,
  `coverage_matrix`. Zostawione `aggregate_by_repetition_group` + dodane `group_means`.
- [x] `components/theme.py` (NOWY) — `apply_house_style` (paleta, siatka hairline, legenda
  h; przy facetach legenda na dole), `mark_whole_system_traces` (szrafura bar+box),
  `ARCH_CLASS_ORDER_COLORS`, `device_color_map`, `ordered_device_types`, `add_reference_line`.
- [x] `components/narrative.py` (NOWY) — `interpretation(co_pokazuje, jak_czytac,
  co_wynika, hipoteza=None)` jako `st.expander` (domyślnie rozwinięty).
- [x] `components/page.py` (NOWY) — `load_runs_or_stop`, `header`, `figure_block`
  (wykres + eksport PNG/SVG + narracja), `table_block` (tabela z przewijaniem + eksport),
  `render_device_legend`, `growing_note`.
- [x] `components/charts.py` — przepisane: `bar_metric_by_device` (poziomy, error bary,
  fade small-n, szrafura), `line_metric_vs_batch`, `pareto_scatter` (hue×symbol + front),
  `box_metric_by_device` (poziomy), `coverage_heatmap`, `paired_bars_same_machine`
  (poziomy), `scatter_over_time`, `hbar_metric_by_device`, `grouped_bar_generic`
  (v/h, facety). Każdy → `go.Figure` + placeholder przy braku danych.
- [x] `.streamlit/config.toml` — deterministyczny jasny motyw (wykresy idą do pracy).
- [x] `app.py` — router `st.navigation` (11 `st.Page`), strony w `views/`.
- [x] `views/0_start.py` … `views/10_eksport.py` — wszystkie 11 stron napisane, renderują
  się bez wyjątku (sprawdzone w przeglądarce, port 8532).
- [x] Stare `pages/1_..._5_...py` usunięte.
- [x] Narracja skorygowana pod „KLUCZOWY WYNIK ANALIZY" (str. 1, 3, 4, 7, 9) — bez
  nadinterpretacji, oparte na `same_machine_winners`.
- [x] `data_access/aggregation.py` — dodane `same_machine_winners` (uczciwe porównanie
  w obrębie maszyny), `agg_two_metrics`; `build_decision_table` przełączone na
  `group_col=device_type_label` + filtr `device_types`.
- [x] Testy: `tests/conftest.py` (atrapy z `.range()` + `make_runs_rows`),
  `test_queries.py` (przepisane + test paginacji >1000), `test_i18n.py`,
  `test_enrich.py`, `test_aggregation.py`, `test_format.py`. **47 testów, wszystkie
  zielone** (`./venv/bin/python -m pytest -q`). `pytest` dopisany do
  `requirements/requirements-dashboard.txt`.

- [x] Skill `design:design-critique` — przejrzane zrzuty stron 0–9, naniesione poprawki
  (patrz „Poprawki po krytyce wizualnej" niżej).
- [x] `dashboard/README.md` — przepisane: strony 0–10, `views/` + `st.navigation`,
  Supabase jedyne źródło, rosnący zbiór + maszyny wojtek/miłosz/maciej, system wizualny,
  testy.
- [x] Podsumowanie hipotezy przekazane użytkownikowi (i wpisane na str. 9 + tu wyżej).

### Sesja weryfikacyjna (2026-09-05) — kod z 09-04 uruchomiony i sprawdzony na żywo

- [x] **Walidacja palety `device_type`** przez port Python skryptu skilla
  `dataviz/scripts/validate_palette.js` (node niedostępny w tym środowisku —
  przepisano ten sam algorytm OKLab/Machado-CVD 1:1 na Python, patrz
  `/tmp/.../scratchpad/validate_palette.py`). Wynik dla 6 kolorów `device_type`
  (adjacent pairs, tak jak są używane na słupkach/liniach): **PASS** na wszystkich
  progach. Dla `--pairs all` (tak jak Pareto na str. 3 używa wszystkich 6 naraz):
  CVD separation i normal-vision floor **FAIL** dla par npu(zielony)↔cuda(pomarańcz)
  i intel(magenta)↔cuda — to udokumentowane, nieuniknione ograniczenie tej palety
  skilla przy >3 slotach w trybie all-pairs (patrz `references/palette.md`: „the
  full eight cannot clear the floors [...] no ordering can"). Pareto **już ma**
  mitygację zgodną z metodą skilla: kodowanie kolor+kształt (symbol per
  `device_type`) + legenda z tekstem + jawne zastrzeżenie w bloku „Co pokazuje"
  („sama barwa może nie wystarczyć dla 6 serii, stąd dodatkowo kształt”) — uznane
  za wystarczające, NIE zmieniano kolorów (zmiana kolejności/kolorów i tak nie
  rozwiązuje problemu przy 6 seriach all-pairs, sam skill to potwierdza).
- [x] **Naprawiona realna usterka**: rampa porządkowa `PRECISION_COLORS`
  (`i18n/pl.py`) — walidator (`--ordinal`) wykrył, że najjaśniejszy krok (`int4`,
  rezerwa, nieużywana w danych) miał kontrast 1,29:1 względem tła (próg 2:1).
  Przesunięto całą rampę o krok w głąb skali sekwencyjnej z `palette.md`
  (fp32=`#0d366b` krok700, fp16=`#1c5cab` krok550, int8=`#3987e5` krok400,
  int4=`#86b6ef` krok250) — teraz PASS na wszystkich 4 sprawdzeniach ordinal.
  Zweryfikowano, że żaden test nie hardkoduje starych hexów (`grep` czysty).
- [x] **Naprawiona realna usterka**: `.streamlit/config.toml` — samo `[theme]` w tej
  wersji Streamlit (1.62) pokrywa TYLKO wariant „Light"; wariant „Dark" (wybierany
  automatycznie przez przeglądarkę/system, bez widocznego przełącznika w UI —
  potwierdzone w DevTools: `localStorage['stActiveTheme-/-v2'] === "System"`) oraz
  pasek boczny nawigacji (`[theme.sidebar]`) są OSOBNYMI sekcjami, które nie
  dziedziczą automatycznie z `[theme]` — bez nich dashboard renderował się z
  ciemnym paskiem bocznym / ciemnym motywem przy niektórych ustawieniach
  przeglądarki, mimo że wymóg to „deterministyczny jasny motyw niezależnie od
  ustawień systemu". Dodano jawne `[theme.dark]`, `[theme.sidebar]`,
  `[theme.dark.sidebar]` z tymi samymi kolorami co `[theme]`. Zweryfikowane na
  żywo w przeglądarce (Computed Style: `backgroundColor` sidebar = `#f4f5f7`,
  `color` = `#0b0b0b`, `filter: none` na całym łańcuchu przodków) — DOM jest
  poprawny w 100% przypadków; przy JEDNEJ konkretnej szerokości viewportu (domyślny
  rozmiar tego narzędzia do zrzutów ekranu w tej sesji, ok. odpowiednik ~800px)
  zrzut ekranu pokazywał ciemny pasek mimo poprawnego DOM/CSS — zniknęło to
  natychmiast po zmianie rozmiaru okna (np. 1000×700 lub 1440×900) i utrzymywało
  się po powrocie do „desktop" presetu, więc to efekt narzędzia/breakpointu przy
  bardzo wąskim viewporcie, nie błąd aplikacji. Do potwierdzenia na realnym ekranie
  użytkownika, ale niska szansa wpływu (typowe okno przeglądarki jest szersze).
- [x] **Uruchomiona aplikacja na żywo** (`.claude/launch.json` dodany, port 8501,
  Supabase realne dane: 1263 przebiegi, 6 maszyn, 15 urządzeń, 3 zadania) —
  wszystkie 11 stron (0–10) przeklikane w przeglądarce, zrzuty ekranu zrobione,
  zero wyjątków Streamlit, dane ładują się poprawnie, filtry (precyzja/faza/
  zadanie/maszyna) działają, eksport CSV/XLSX/PNG generuje pliki (potwierdzone
  w Network: 200 OK). Zweryfikowano punktowo zgodność z danymi: str. 5 pokazuje
  dokładnie 1 słupek (fp32) dla NPU/iGPU Intel, 2 słupki (fp32/fp16) dla iGPU AMD/
  ROCm/CUDA, 3 słupki (fp32/fp16/int8) dla CPU — zgodnie ze specyfiką danych.
  Str. 9 pokazuje 136 porównań, CPU=19 (wszystkie na `artur-PC`), CUDA=68,
  ROCm=24-25, iGPU Intel=24 — zgodne z „KLUCZOWYM WYNIKIEM ANALIZY" wyżej.
- [x] **Konsola przeglądarki**: jedyne błędy to 20× `404` na
  `/<nazwa_strony>/_stcore/health` i `/<nazwa_strony>/_stcore/host-config` (po 2 na
  każdą z 10 nie-głównych stron) — Streamlit w tej wersji buduje ten request
  względem ścieżki `st.navigation` zamiast względem roota, po czym NATYCHMIAST
  ponawia poprawnie pod `/_stcore/health` (200 OK) — kosmetyczny szum w DevTools,
  zero wpływu funkcjonalnego/wizualnego, nie do naprawienia z poziomu kodu
  aplikacji (wewnętrzne zachowanie frontendu Streamlit).

## W trakcie

- Nic. Zadanie zrealizowane i zweryfikowane na żywo (sesja 2026-09-05).

## TODO (opcjonalne / do rozważenia)

- `dashboard/venv` vs `requirements` — `pandas 3.0.5` / `plotly 7.0.0` są nietypowo nowe;
  jeśli CI używa starszych, sprawdzić `error_x/error_y` w px.bar i `st.navigation`.
- Drobna kosmetyka: na str. 5 prawy wykres (accuracy) powiela etykiety osi Y lewego —
  można je ukryć. Nie blokuje.
- `line_metric_vs_batch`: pionowe linie przecięć CPU↔inne bywają mało widoczne pod
  danymi — rozważyć adnotację zamiast/obok `add_vline`.
- **Nic nie jest jeszcze scommitowane** — `git log -- dashboard/` pokazuje tylko
  commit sprzed przebudowy (`7e7d1a1b`). Użytkownik nie prosił o commit w tej
  sesji, więc zmiany zostały w working tree (`git status` pokaże pliki `M`/`??`
  zgodnie z listą w tym pliku) — do zrobienia jawnie na życzenie użytkownika.
- `.claude/launch.json` (root repo) dodany w tej sesji WYŁĄCZNIE do podglądu
  dashboardu w przeglądarce (`streamlit run` z `cd dashboard` + jawny port 8501) —
  to narzędzie deweloperskie, nie część dashboardu; można zostawić lub usunąć.

## Poprawki po krytyce wizualnej (`design:design-critique`, 2026-09-04)

Przejrzane zrzuty stron 0–9. Naniesione:
- **Słupki poziome** wszędzie, gdzie oś kategorii to `device_type`/architektura/scenariusz
  (długie nazwy „NPU Intel (OpenVINO)" itp. rotowane -20° w wąskich panelach były
  nieczytelne). Dotyczy: `bar_metric_by_device`, `box_metric_by_device`,
  `paired_bars_same_machine`, `grouped_bar_generic` (opcja `orientation="h"`),
  wykres CV na str. 8.
- **Legenda na dole** przy wykresach z facetami (kolidowała z paskami tytułów paneli) —
  `apply_house_style` wykrywa `>1` osi X.
- **Zbite tytuły osi** przy facetach — jeden, wyśrodkowany tytuł X + jeden pionowy Y
  (px powielał je pod każdym panelem). Usunięty też surowy „device_type_label" jako
  etykieta osi.
- **Log-oś rozjeżdżała się** na scatterze software-vs-watomierz (str. 8): `zeroline`
  na osi log celuje w log(0) = -inf; wyłączone + jawny `range` w log10. Dodatkowo
  `add_identity_line` jako *trace* (nie `shape`, bo shape na log-osi wymaga log10).
- **`smart_plug == 0.0`** (20 wierszy `intel_gpu_openvino`) → NaN w `enrich` (to brak
  odczytu, nie realne zero) + zerowanie `power_discrepancy_pct` bez odczytu watomierza.
- **Polskie etykiety paneli** (`task`/`phase`) — `_facet_labels` tnie prefiks „kol=".
- **`go.Box` nie wspiera `fillpattern`** → wywalało str. 6; szrafurę boxów zastąpiła
  legenda kolorów z gwiazdką + podpis.
- **Puste kategorie na osi** (str. 9 „iGPU AMD" z 0 wygranych) — `category_orders`
  ograniczone do obecnych.
- **`build_decision_table` pooled zawsze wskazywał CUDA** (uśrednia 2 energooszczędne
  laptopy) → str. 9 przełączona na `same_machine_winners` (uczciwe porównanie w obrębie
  maszyny). Pooled zostaje jako pomocnicza tabela na str. 10.
- **Narracje** przepisane pod faktyczny wynik (patrz „KLUCZOWY WYNIK ANALIZY").

Świadomie zostawione (nie blokuje, w TODO opcjonalnym): powielone etykiety osi Y prawego
wykresu na str. 5; słaba widoczność pionowych linii przecięć na str. 4.

## Plan stron (0–10) — plan pierwotny; IMPLEMENTACJA w `src/dashboard/views/N_*.py`, nie `pages/`

- **0 Start / Jak czytać** (`app.py` → `views/0_start.py`): akapit o hipotezie, legenda kolorów `device_type`,
  `device_only` vs `whole_system` (co mierzy każde źródło, watomierz = całe gniazdko),
  KPI (przebiegi, maszyny, urządzenia, zadania, zakres dat), heatmapa pokrycia
  `device_type × task × phase`, dopisek „zbiór rośnie".
- **1 Efektywność energetyczna** (`pages/1_...`): J/próbkę wg `device_type`, panele
  `task × phase`, słupki + error bary, przełącznik `precision`; wariant „próbki/dżul".
- **2 Wydajność** (`pages/2_...`): `throughput [próbki/s]` i `duration_s` wg `device_type`.
- **3 Kompromis Pareto** (`pages/3_...`): scatter `throughput` X vs `J/próbkę` Y(log),
  kolor+symbol = `device_type`, panel = `task`, front Pareto.
- **4 Wpływ batch size** (`pages/4_...`): J/próbkę i throughput vs `batch_size`, linie per
  `device_type`, panel per `task`/`model`; zaznaczyć przecięcia CPU↔GPU.
- **5 Wpływ precyzji** (`pages/5_...`): J/próbkę i `accuracy` obok siebie per `device_type`;
  `int8`=tylko CPU, `intel/npu`=tylko fp32; `quantization_status`. Mini-sekcja
  `llm_inference` (eksploracyjnie).
- **6 Moc i temperatura** (`pages/6_...`): box/violin `avg/peak_power_watts` per
  `device_type`, `peak` vs `avg`, `avg_temperature_c` gdzie jest.
- **7 CPU vs akcelerator (ta sama maszyna)** (`pages/7_...`): pary słupków J/próbkę +
  „ile razy oszczędniej/drożej", per maszyna, budowane dynamicznie.
- **8 Co mierzymy / powtarzalność** (`pages/8_...`): software vs watomierz Tuya,
  `power_discrepancy_pct` z komentarzem „różnica zakresu, nie błąd"; nacisk na
  `coefficient_of_variation_energy` per `device_type`/`task`; próbki mocy vs próg;
  `measurement_scope`; anomalia `flops_per_watt` ROCm.
- **9 Weryfikacja hipotezy / wnioski** (`pages/9_...`): tekst + tabela decyzyjna
  (`task, model, phase, batch_size` → zwycięzca J/próbkę + o ile + odsyłacz). Zastrzec
  serie `whole_system`.
- **10 Eksport danych** (`pages/10_...`): jak obecna str. 5 + eksport tabeli decyzyjnej.

## Decyzje projektowe

- **`llm_inference` poza głównymi porównaniami** — `num_iterations` skrajnie zmienne
  (308–4927), brak realnych grup powtórzeń (n=1/grupa), CV energii ogromne; tylko CPU.
  Trafia do osobnej mini-sekcji eksploracyjnej na str. 5, wykluczone z tabeli decyzyjnej
  (str. 9) i z Pareto (str. 3).
- **Dwa poziomy grupowania:** `device_type` (6-wart.) = kolor wszędzie; `arch_class`
  (4-wart.: CPU / GPU dyskretne / iGPU / NPU) = grupowanie/panelowanie i tabela decyzyjna.
- **NPU jako osobny `arch_class`** (akcelerator dedykowany, praca §1.2) — nie „GPU".
- **`whole_system` (amd_igpu, intel_gpu_openvino, npu_openvino) vs `device_only`
  (cpu, cuda, rocm)** — serie `whole_system` szrafowane (`pattern_shape`) + dopisek
  „cały SoC / komputer", osobna grupa legendy. Nie zestawiać wprost bez zastrzeżenia.
- **Paleta:** dataviz — sloty 1–6 w stałej kolejności: cpu=blue `#2a78d6`,
  cuda=orange `#eb6834`, rocm=aqua `#1baf7a`, amd_igpu=yellow `#eda100`,
  intel_gpu_openvino=magenta `#e87ba4`, npu_openvino=green `#008300`. Na Pareto (all-pairs)
  dodatkowo symbol per `device_type` (kodowanie złożone hue×kształt) + twin tabela.
- **Skala log** dla energii/próbkę i throughput (rozrzut ~100–300×).
- **Small n:** grupy z `n<3` (obecnie ~35% grup ma n=2) — słupki półprzezroczyste +
  adnotacja; w tooltipie zawsze `n`.
- **Numeracja stron:** `app.py` = strona 0; `pages/1_...`–`pages/10_...`. Stare pliki
  `pages/1_Ranking_...`..`5_Eksport_...` zostaną zastąpione.

## Znane problemy / do weryfikacji

- `flops_per_watt` ROCm systematycznie ~4–5× niższe niż CUDA przy podobnej pracy
  (image inf: rocm 1,3·10⁷ vs cuda 4,6·10⁷; nlp: rocm 1,35·10⁸ vs cuda 6,3·10⁸) —
  opisać jako „do weryfikacji" (możliwy artefakt szacowania FLOPs/mocy dla ROCm).
- `power_discrepancy_pct` tylko tam, gdzie był watomierz: `cpu` (~141 w.) i `rocm`
  (~114 w.); duże wartości (śr. ~40–70%) — efekt zakresu (gniazdko vs układ), nie błąd.
- `power_samples_count < 20` dla ~197/1263 przebiegów — mniej wiarygodne avg/peak power.
- `quantization_status`: 112 `partial` vs 36 `success` (int8 przeważnie tylko warstwy Linear).
- 1 „osierocony" `repetition_group_id` z n=1.
- pandas 3.0.5 / plotly 7.0.0 / kaleido 1.4.0 w `dashboard/venv` — nietypowo nowe,
  uważać na zmiany API (downcasting, `groupby`, eksport kaleido v1).

## Charakterystyka danych (zweryfikowana w Supabase, 2026-09-03)

- **runs:** 1264 (1263 z `run_summary`), wszystkie `status='completed'`. Zakres dat
  `started_at`: 2026-08-20 → 2026-09-03.
- **Maszyny (6, lista rośnie):** artur-MS-7B86 (cpu, intel_gpu_openvino, npu_openvino),
  artur-PC (cpu, rocm), business-dell (cpu, intel_gpu_openvino),
  kamila-galaxybook2pro (cpu, intel_gpu_openvino), ola (cpu, cuda, amd_igpu),
  proksza-legion-slim5 (cpu, cuda, amd_igpu). Brak jeszcze wojtek / miłosz / maciej.
- **device_type → liczba przebiegów:** cpu 725, cuda 236, rocm 118, amd_igpu 104,
  intel_gpu_openvino 60, npu_openvino 20.
- **measurement_scope:** cpu/cuda/rocm = `device_only`; amd_igpu/intel_gpu_openvino/
  npu_openvino = `whole_system` (NIE `npu_only`).
- **Modele:** image_classification → mobilenet_v3, resnet50; nlp_sentiment →
  distilbert-base-uncased; llm_inference → tiny-gpt2.
- **Batch size:** image_classification 1/4/32/64; nlp_sentiment 1/4/8/16; llm_inference 1.
- **phase=train** tylko cpu/cuda/rocm. amd_igpu/intel/npu → tylko inference.
- **Precyzje:** cpu: fp16/fp32/int8; cuda: fp16/fp32 (BRAK int8); rocm: fp16/fp32;
  amd_igpu: fp16/fp32; intel_gpu_openvino: tylko fp32; npu_openvino: tylko fp32.
- **Repetycje:** 477 grup; runs/grupę: 169× n=2, 308× n=3, 1× n=1. Brak n≥4.
- **CV energii:** ~53% grup < 0,05 (bardzo powtarzalne); ogon ~13% grup > 0,30.
- **Rzędy wielkości J/próbkę (image inf):** cuda ~0,029 / intel ~0,033 / npu ~0,037 /
  amd_igpu ~0,26 / rocm ~0,32 / cpu ~0,34 (śr.), ale min: cpu 0,0017 vs cuda 0,0012 —
  przy części konfiguracji CPU praktycznie dorównuje.
- **avg_power_watts (image inf, śr.):** cuda 17,7 / intel 21,3 / npu 30,9 / amd_igpu 31,5 /
  cpu 50,2 / rocm 62,5. rocm nlp/train: ~170–200 W (cała karta, ROCm-SMI).
- **accuracy:** 0,2 (train 1 epoka) – 0,73 (nlp train); image inf ~0,44. Tylko sanity-check.
- **smart_plug (watomierz):** rocm 114/118, cpu 141/725, intel_gpu 20/60; cuda/amd_igpu/npu 0.
