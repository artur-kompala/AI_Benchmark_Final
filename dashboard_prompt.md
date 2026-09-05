# Prompt do Claude — przebudowa i mocne ulepszenie istniejącego dashboardu Streamlit

> Wklej całość poniżej linii jako jedną wiadomość do Claude, uruchomionego w tym repo
> (musi mieć dostęp do `dashboard/` oraz `.env` z danymi do Supabase).

---

## Zadanie

Mam **działający dashboard w Streamlit** w `dashboard/` (analiza wyników benchmarku do
mojej pracy magisterskiej). Jest **nieczytelny, zagracony i słabo prowadzi czytelnika
przez wnioski**. Przebuduj go i mocno ulepsz — **zmiany mogą być radykalne**: nowa
struktura stron, nowe wykresy, nowy układ, przepisane komponenty. Zachowaj tylko to, co
poniżej oznaczone jako „nie ruszać".

Cel: **dużo prostych, czytelnych wykresów**, pogrupowanych w **więcej zakładek
tematycznych**, gdzie **pod każdym wykresem** jest zwięzły opis w trzech blokach:
**Co pokazuje / Jak czytać / Co z tego wynika** (z odniesieniem do hipotezy pracy).
Dashboard ma prowadzić za rękę do wniosku, a nie wysypywać surowe wykresy.

### Kolejność pracy (obowiązkowa)

1. **Najpierw wczytaj skill `dataviz`** — zanim napiszesz pierwszą linię kodu wykresu,
   dobierzesz kolory albo ułożysz kafelki KPI. Z niego bierz: paletę (odporną na
   daltonizm), reguły osi/legend/tooltipów, dobór typu wykresu, układ dashboardu,
   formatowanie liczb.
2. Nie ma dedykowanego skilla „data analytics" — **analizę zrób rzetelnie ręcznie**:
   poprawna agregacja po grupach powtórzeń, słupki błędów z odchylenia std., współczynnik
   zmienności (CV), fronty Pareto, porównania parami w obrębie tej samej maszyny, jawne
   oznaczanie małego `n`. Nie uśredniaj byle czego bez podania `n`.
3. Po przebudowie: uruchom aplikację, zrób zrzuty kluczowych stron i przejdź skillem
   **`design:design-critique`** po tych zrzutach — nanieś poprawki z krytyki.

### Dziennik postępu — `IMPLEMENTATION_STATUS.md` (prowadź na bieżąco)

Na początku pracy utwórz w korzeniu repo plik `IMPLEMENTATION_STATUS.md` i **aktualizuj
go po każdym znaczącym kroku** (nowy/zmieniony plik, gotowa strona, podjęta decyzja
projektowa, dodany test, napotkany problem). To zabezpieczenie na wypadek przerwania
sesji (koniec kredytów / kontekstu) — **kolejna sesja ma móc wznowić pracę wyłącznie na
podstawie tego pliku**, bez zgadywania i bez czytania całej historii rozmowy.

Struktura:
- **Stan ogólny** — 2–3 zdania: co działa, co jest w trakcie.
- **Zrobione** — konkretnie (plik, funkcja, strona), odhaczone.
- **W trakcie** — co dokładnie jest teraz otwarte i w którym pliku.
- **TODO** — kolejne kroki po kolei, na tyle konkretne, że da się je podjąć „na zimno".
- **Decyzje projektowe** — z uzasadnieniem (np. „`llm_inference` poza głównymi
  porównaniami, bo…", „NPU jako osobny `device_type`", wybór typów wykresów).
- **Znane problemy / odłożone** — błędy, obejścia, rzeczy do weryfikacji.

Zasady: dopisuj przyrostowo (nie przepisuj całości), datuj wpisy, odwołuj się do plików
ścieżką. Ten plik jest **źródłem prawdy o postępie** — czego w nim nie ma, uznajemy za
niezrobione. Jeśli `IMPLEMENTATION_STATUS.md` już istnieje na starcie, najpierw go
przeczytaj i kontynuuj od sekcji „W trakcie" / „TODO".

## Kontekst merytoryczny (praca magisterska)

Temat: **„Analiza energooszczędności i wydajności nowoczesnych układów obliczeniowych w
zastosowaniach sztucznej inteligencji"**. Porównanie CPU vs GPU dyskretne vs iGPU vs NPU
(akcelerator dedykowany, praca §1.2) pod kątem kompromisu wydajność ↔ zużycie energii
przy zadaniach ML **w warunkach ograniczonych zasobów** (laptopy, domowe PC — nie centra
danych).

**Hipoteza do zweryfikowania na dashboardzie:** efektywność energetyczna istotnie zależy
od architektury (CPU/GPU) i parametrów wykonania; **dla modeli o umiarkowanej złożoności
i małych rozmiarach partii (batch size) CPU może mieć porównywalną lub wyższą
efektywność energetyczną niż GPU, mimo niższej maksymalnej wydajności.** Metryki
rozstrzygające: **energia na próbkę [J/próbkę]** oraz energia na całe zadanie; sedno to
zależność czas ↔ moc chwilowa ↔ energia sumaryczna.

Dashboard ma dać wizualną odpowiedź: **kiedy GPU jest uzasadnione również energetycznie,
a kiedy CPU jest wyborem racjonalniejszym.**

## Co już jest (stan wyjściowy w `dashboard/`)

Streamlit multipage, Python 3.11, `plotly` + `kaleido` (eksport PNG/SVG),
`pandas`, czyta **na żywo z Supabase** przez `supabase-py` — **Supabase jest jedynym
źródłem danych** (bez trybu offline i bez importu z plików).

```
src/dashboard/
  app.py                         # strona główna: 5 kafelków KPI + tabela ostatnich przebiegów
  i18n/pl.py                     # etykiety PL, mapy kategorii urządzeń, device_category_label()
  data_access/
    supabase_client.py           # klient read-only z .env (SUPABASE_URL/KEY)
    queries.py                   # get_runs_df() -> spłaszczony DF (runs + devices(*) + run_summary(*)),
                                 #   get_run_stats_summary_df(), get_devices_df(), *_cached (ttl=60)
    aggregation.py               # aggregate_by_repetition_group(): mean/std/n po repetition_group_id
  components/
    charts.py                    # ~10 funkcji plotly, paleta Okabe-Ito, DEVICE_CATEGORY_COLORS/SYMBOLS
    export.py / export_widgets.py# eksport PNG >=300 DPI / SVG / CSV / Excel, opisowe nazwy plików
  pages/
    1_Ranking_Energooszczednosci.py    # bar J/próbkę per urządzenie + error bary + tabela
    2_Wplyw_Parametrow.py              # linie J/próbkę vs batch_size (per precyzja, facet per urządzenie) + panel INT8
    3_Wiarygodnosc_Pomiaru.py         # power_discrepancy_pct scatter/bar + liczba próbek mocy + measurement_scope
    4_GPU_Dyskretne_vs_Zintegrowane.py# tylko GPU, podział dyskretne/zintegrowane
    5_Eksport_Danych.py               # pełne tabele + eksport CSV/Excel
tests/  test_queries.py test_export.py   # offline, z atrapą klienta Supabase
```

### Nie ruszać (albo tylko rozszerzać, nie usuwać)
- **Mechanizm eksportu** `components/export*.py` — PNG/SVG ≥300 DPI + CSV/Excel z
  opisową nazwą pliku. Każdy nowy wykres i każda nowa tabela **musi** mieć ten eksport
  (wykresy trafiają wprost do pracy — to twardy wymóg).
- **Warstwa read-only** — dashboard **nigdy** nie pisze do Supabase ani nie uruchamia
  benchmarków (to `benchmark_runner`, osobny proces).
- **Jedno źródło danych: Supabase na żywo** — nie dodawaj trybu offline, importu z CSV
  ani lokalnego cache'u plikowego. Dane wciąż przyrastają (trwają testy na kolejnych
  maszynach), więc dashboard ma zawsze pokazywać aktualny stan z Supabase.
- **Polskie UI** przez `i18n/pl.py` — rozszerzaj ten moduł, nie rozsypuj stringów po kodzie.
- **Zdegradowany stan zamiast wyjątku** przy braku danych/tabeli/migracji — zachowaj.
- Terminologia spójna z pracą: *moc chwilowa*, *energia całkowita*, *energia na próbkę*,
  *energooszczędność*.

## Dane

### Źródło: Supabase (na żywo, jedyne)
Dashboard czyta wyłącznie z Supabase przez `data_access/queries.py` (`get_runs_df()` →
spłaszczony DataFrame: `runs` + `devices(*)` + `run_summary(*)`, plus
`get_run_stats_summary_df()`, `get_devices_df()`, wersje `*_cached` z `ttl=60`). Bez
trybu offline, bez loadera plikowego, bez przełącznika źródła. **Dane są w trakcie
zbierania** — liczby przebiegów poniżej to stan chwilowy i będą rosły w miarę napływania
wyników z kolejnych maszyn (patrz „Maszyny testowe"). Nie hardkoduj liczby przebiegów,
maszyn ani kompletu kombinacji — wszystko wyliczaj z danych przy każdym odświeżeniu.

### Tabele w Supabase
| tabela | wierszy (stan chwilowy, rośnie) | rola | klucz |
|---|---|---|---|
| `devices` | ~13 | wymiar sprzętowy | `id` |
| `runs` | ~1200+ | macierz eksperymentu, 1 wiersz/przebieg | `id`, `device_id`→devices |
| `run_summary` | 1:1 z `runs` | policzone metryki | `run_id`→runs.id |
| `run_stats_summary` | ~470+ | agregaty po grupie powtórzeń (mean/std/CV) | `repetition_group_id` |

**Kolumny kluczowe:**
- `runs`: `task` (`image_classification`/`nlp_sentiment`/`llm_inference`), `model_name`
  (`mobilenet_v3`, `resnet50`, `distilbert-base-uncased`, `tiny-gpt2`), `phase`
  (`train`/`inference`), `batch_size`, `precision` (`fp32`/`fp16`/`int8`),
  `measurement_scope` (`device_only`/`whole_system`), `duration_s`,
  `throughput_samples_per_s`, `avg_temperature_c` (część wierszy), `repetition_group_id`,
  `config_snapshot` (JSON, ma `experiment_name`), `status` (wszystkie `completed`).
- `run_summary`: `energy_joules_software`, `energy_joules_smart_plug`,
  `power_discrepancy_pct` (część wierszy — tam gdzie był watomierz), `energy_per_sample_joules`,
  `energy_per_epoch_joules`, `avg_power_watts`, `peak_power_watts`, `flops_per_sample`,
  `flops_per_watt`, `samples_processed`, `quantization_status` (`''`/`partial`/`success`),
  `accuracy` (większość wierszy, ~0,13–0,83), `power_samples_count`.
- `run_stats_summary`: `n_repetitions`, `mean_duration_s`, `std_duration_s`,
  `mean_energy_joules`, `std_energy_joules`, `coefficient_of_variation_energy`.
  **To źródło słupków błędów** — dla energii CAŁKOWITEJ. Dla energii NA PRÓBKĘ agreguj
  jak w `data_access/aggregation.py` (mean/std po `repetition_group_id`).

### Maszyny testowe (lista rośnie — testy w toku)
Stan obecny:
| machine_name | CPU | GPU dyskretne | iGPU | NPU |
|---|---|---|---|---|
| `artur-PC` | Ryzen 7 5700X3D | Radeon RX 7800 XT (`rocm`) | – | – |
| `artur-MS-7B86` | Core Ultra 7 155H | – | Arc iGPU (`intel_gpu_openvino`) | Intel AI Boost (`npu_openvino`) |
| `business-dell` | Core i5-1335U | – | Iris Xe (`intel_gpu_openvino`) | – |
| `kamila-galaxybook2pro` | Core i7-1260P | – | Iris Xe (`intel_gpu_openvino`) | – |
| `ola` | Ryzen 7 5800HS | RTX 3050 Ti Laptop (`cuda`) | Radeon Vega (`amd_igpu`) | – |
| `proksza-legion-slim5` | Ryzen 7 7840HS | RTX 4060 Laptop (`cuda`) | Radeon 780M (`amd_igpu`) | – |

W drodze (wyniki dojdą do Supabase; nazwy maszyn pojawią się same z danych — **nie
hardkoduj listy**):
- **wojtek** — CPU + GPU dyskretne AMD (`rocm`).
- **miłosz** — CPU + GPU dyskretne NVIDIA (`cuda`).
- **maciej** — CPU + NPU (`npu_openvino`) + iGPU.

Dashboard ma budować listę maszyn i urządzeń dynamicznie z zapytania, tak żeby nowe
maszyny i nowe przebiegi pojawiały się bez zmian w kodzie.

### Charakterystyka danych — trzymaj się jej, nie zgaduj
- **Dane wciąż przyrastają** — liczby przebiegów poniżej to stan chwilowy, nie limit.
- Rozmiary partii: `image_classification` → 1/4/32/64; `nlp_sentiment` → 1/4/8/16;
  `llm_inference` → tylko 1.
- Faza `train` tylko na `cpu`/`cuda`/`rocm`. iGPU (`amd_igpu`, `intel_gpu_openvino`) i
  NPU (`npu_openvino`) — **tylko `inference`**.
- `intel_gpu_openvino`: tylko `image_classification` + `inference` + `fp32`.
- `npu_openvino`: tylko `image_classification` (`mobilenet_v3`, `resnet50`) + `inference`
  + `fp32`. To **trzecia architektura** obok CPU i GPU (akcelerator dedykowany, praca
  §1.2) — traktuj jako osobny `device_type`, nie wciągaj do „GPU".
- `int8` praktycznie tylko CPU (dynamiczna kwantyzacja PyTorch = CPU) — celowe, zgodne z pracą.
- `llm_inference`: tylko CPU, tylko `tiny-gpt2`, `num_iterations` skrajnie
  zmienne → **osobna mała sekcja eksploracyjna albo wyklucz z głównych porównań** (zdecyduj,
  napisz dlaczego).
- **`measurement_scope`:** `cpu`/`cuda`/`rocm` = `device_only` (pobór samego układu);
  `amd_igpu`, `intel_gpu_openvino` i `npu_openvino` = `whole_system` (pobór całego
  pakietu/SoC albo całego komputera — nie samego układu). **Nie zestawiaj J/próbkę serii
  `whole_system` bezpośrednio z `device_only` bez wyraźnego zastrzeżenia** — oznacz te
  serie wizualnie (szrafura + dopisek „cały SoC / cały komputer") i trzymaj w osobnej
  legendzie. To zastrzeżenie jest wprost w pracy.

### Błędy w obecnym kodzie — napraw przy okazji
- `i18n/pl.py::device_category_label()` **nie obsługuje `amd_igpu`** → wpada do
  `device_type or "?"` i tworzy śmieciową kategorię „amd_igpu". Powinno mapować na
  **„GPU zintegrowane (iGPU)"**.
- Strony 4 i wykres `gpu_discrete_vs_integrated_chart` filtrują do
  `["cuda","rocm","intel_gpu_openvino"]` i **pomijają `amd_igpu`** — dołącz go.
  `npu_openvino` **nie** wchodzi na ten wykres (to nie GPU) — pokaż go w wykresach per
  `device_type`.
- NPU (`npu_openvino`) ma już wyniki w bazie — wepnij go na stałe w mapę kolorów/symboli
  `device_type`, w `device_category_label()` (→ np. „NPU (akcelerator dedykowany)") i we
  wszystkie wykresy per `device_type`. Uwaga: przebiegi NPU mają `measurement_scope =
  whole_system` (nie `npu_only`) — etykieta scope ma to odzwierciedlać.
- Ogólna zasada dla kombinacji, których naprawdę jeszcze nie ma w danych: ukrywaj
  kategorię/panel gdy `n = 0`, zamiast rysować pustą serię.

### Anomalie — pokaż, nie ukrywaj
- `flops_per_watt`: `rocm` ≈ 5,7·10⁷ vs `cuda` ≈ 2,7·10⁸ vs `amd_igpu` ≈ 2,7·10⁸ —
  niespójne. Opisz jako „do weryfikacji" (możliwy artefakt szacowania FLOPs/mocy dla
  ROCm), bez mocnych wniosków.
- `power_discrepancy_pct` bywa duże, ale **w większości wynika z różnicy zakresu
  pomiaru**, nie z błędu: watomierz Tuya mierzy pobór **całego komputera z gniazdka**
  (zasilacz, ekran, płyta, straty), a pomiar programowy — sam układ / SoC. Przy seriach
  `device_only` różnica jest więc spodziewana i systematyczna. Pokaż to jako zestawienie
  „co dokładnie mierzy każde źródło", a nie jako podważenie wiarygodności — nie strasz
  liczbą, opisz przyczynę. Realnym sygnałem jakości jest raczej
  `coefficient_of_variation_energy` (powtarzalność powtórzeń).
- `accuracy` bywa niska (`train` przy 1 epoce, kwantyzacja) — sanity-check, nie wniosek o
  jakości modelu.

## Wymagana przebudowa

### Warstwa danych
- Zostaw jedno źródło: Supabase na żywo (`data_access/queries.py`, `*_cached` z ttl).
  **Nie** dodawaj loadera plikowego ani przełącznika źródła.
- Dołóż kolumny pochodne raz, centralnie: `energy_per_1000_samples_j`, `samples_per_joule`
  (= odwrotność J/próbkę, „im wyżej tym lepiej"), `is_whole_system` (bool), czytelny
  `device_short` (np. „RTX 4060 (cuda)", „i7-1260P (cpu)", „Intel AI Boost (npu_openvino)").
- Napraw `device_category_label` i wepnij `npu_openvino` (patrz wyżej).
- Listę maszyn, urządzeń i kombinacji wyliczaj z danych przy każdym odświeżeniu —
  dojdą nowe maszyny (wojtek / miłosz / maciej) i nowe przebiegi.

### Komponent narracji (nowy, używany wszędzie)
`components/narrative.py::interpretation(co_pokazuje, jak_czytac, co_wynika)` renderujący
trzy zwięzłe bloki pod wykresem (np. `st.markdown` z pogrubionymi nagłówkami albo
`st.expander` domyślnie rozwinięty). **Każdy wykres w dashboardzie** ma taki blok.

### Strony (proponowana nowa struktura — możesz doprecyzować, zachowaj sens i kolejność)
0. **Start / Jak czytać ten dashboard** — 1 akapit o hipotezie (CPU vs GPU vs NPU jako
   trzecia architektura), legenda kolorów `device_type`, wyjaśnienie `device_only` vs
   `whole_system` (co dokładnie mierzy każde źródło, w tym watomierz = cały komputer z
   gniazdka), kafelki KPI (przebiegi, maszyny, urządzenia, zadania, zakres dat), heatmapa
   pokrycia `device_type × task × phase` (liczba przebiegów — od razu widać, gdzie są
   luki i które maszyny jeszcze nie dosłały wyników). Dopisek, że zbiór danych rośnie.
1. **Efektywność energetyczna** — `J/próbkę` wg `device_type`, panele `task × phase`,
   słupki + słupki błędów, przełącznik `precision`. Wariant „próbki/dżul" dla intuicji
   „więcej = lepiej". Rdzeń hipotezy.
2. **Wydajność** — `throughput [próbki/s]` i `duration_s` wg `device_type`, te same
   panele. Pokazać przewagę GPU w surowej szybkości (kontrast do str. 1).
3. **Kompromis wydajność ↔ energia (Pareto)** — scatter `throughput` (X) vs `J/próbkę`
   (Y, log), kolor = `device_type`, panel = `task`, zaznaczony front Pareto. Wprost:
   „szybciej ≠ oszczędniej".
4. **Wpływ batch size** — `J/próbkę` i `throughput` vs `batch_size` (linie per
   `device_type`), panel per `task`/`model`. **Kluczowy test hipotezy** — zaznacz punkty
   przecięcia krzywych CPU i GPU.
5. **Wpływ precyzji (fp32 / fp16 / int8)** — `J/próbkę` i `accuracy` obok siebie per
   `device_type`; zaznacz, że `int8` = tylko CPU, a `intel_gpu_openvino` i `npu_openvino`
   = tylko `fp32`; pokaż `quantization_status`.
6. **Moc chwilowa i temperatura** — rozkłady `avg_power_watts` / `peak_power_watts` per
   `device_type` (box/violin), `peak` vs `avg`, `avg_temperature_c` gdzie jest.
7. **CPU vs akcelerator na tej samej maszynie** — najuczciwsze porównanie parami w
   obrębie maszyny: `ola` (CPU vs RTX 3050 Ti vs Vega), `proksza-legion-slim5` (CPU vs
   RTX 4060 vs 780M), `artur-PC` (CPU vs RX 7800 XT), `artur-MS-7B86` (CPU vs Arc iGPU vs
   NPU), maszyny Intel (CPU vs Xe). Pary słupków `J/próbkę` + wskaźnik „ile razy
   oszczędniej/drożej". Buduj pary dynamicznie z danych — dojdą maszyny wojtek / miłosz /
   maciej (ta ostatnia z NPU).
8. **Co dokładnie mierzymy / powtarzalność** — zestaw obok siebie, czym jest każde
   źródło: pomiar programowy (`energy_software`, sam układ / SoC) vs watomierz Tuya
   (`energy_smart_plug`, **cały komputer z gniazdka**). `power_discrepancy_pct` pokaż z
   komentarzem, że różnica to głównie efekt różnego zakresu (gniazdko vs układ), nie
   błąd. Główny nacisk na `coefficient_of_variation_energy` (powtarzalność powtórzeń) per
   `device_type`/`task`, liczbę próbek mocy vs próg, jawne wyjaśnienie
   `measurement_scope`, oraz anomalię `flops_per_watt` ROCm. (Rozszerz obecną stronę 3.)
9. **Weryfikacja hipotezy / wnioski** — tekst + **tabela decyzyjna**: dla każdego
   scenariusza (`task`, `model`, `phase`, `batch_size`) → zwycięzca po `J/próbkę`
   (CPU / GPU dyskretne / iGPU / NPU) i o ile, z odsyłaczem do zakładki. Wprost: kiedy CPU
   dorównuje/bije GPU energetycznie i czy to potwierdza hipotezę. Zastrzeż serie
   `whole_system` (iGPU, NPU) — inny zakres pomiaru niż `device_only`.
10. **Eksport danych** — jak obecna strona 5 (zachowaj), dołóż eksport tabeli decyzyjnej
    ze str. 9.

### System wizualny
- Paleta i reguły z `dataviz`. **Jedna mapa kolorów `device_type`** (`cpu`, `cuda`,
  `rocm`, `amd_igpu`, `intel_gpu_openvino`, `npu_openvino`) w **każdym** wykresie —
  rozszerz `DEVICE_CATEGORY_COLORS`/`SYMBOLS` albo przejdź na klucz po `device_type`.
- Każda oś podpisana **z jednostką** (`J/próbkę`, `W`, `próbki/s`, `°C`, `s`). Skala
  **logarytmiczna** tam, gdzie rozrzut jest rzędu wielkości (energia CPU vs CUDA bywa ~100×).
- Słupki błędów wszędzie, gdzie jest `std`; przy `n < 3` — wizualne oznaczenie
  (przezroczystość / adnotacja).
- Serie `whole_system` (iGPU, NPU) wizualnie odróżnione (szrafura/wzór) i opisane
  „cały SoC / cały komputer".
- Czytelne domyślne: większa czcionka, oddech w marginesach i między panelami facet,
  krótkie etykiety osi X (użyj `device_short`), tooltip z wartością + `n`.
- **Zero chartjunk:** bez 3D, bez wykresów kołowych, bez podwójnej osi Y bez potrzeby.
- Liczby formatowane po polsku, rozsądna liczba miejsc po przecinku.
- `st.set_page_config(layout="wide")` na każdej stronie; szerokie tabele w kontenerze z
  przewijaniem.

### Guardraile
- **Nie zmyślaj wartości.** Brakująca kombinacja = jawna luka („brak danych"), nie
  interpolacja. Przy każdej wartości zagregowanej pokaż `n`.
- Nie mieszaj `whole_system` z `device_only` w jednej serii bez etykiety.
- Zachowaj zdegradowany stan zamiast wyjątku przy brakach.

### Testy
- Rozszerz `tests/` o: poprawna klasyfikacja `amd_igpu` → „GPU zintegrowane (iGPU)" i
  `npu_openvino` → „NPU (akcelerator dedykowany)", agregacja po grupie powtórzeń, budowa
  tabeli decyzyjnej, dynamiczne wyliczanie listy maszyn/urządzeń z DataFrame. Testy
  offline (bez sieci), na atrapie klienta Supabase, wzorem obecnych.

## Na koniec
1. Uruchom dashboard (`dashboard/venv`, `streamlit run src/dashboard/app.py`, źródło
   danych = Supabase na żywo z `.env`), sprawdź że każda strona renderuje się bez błędu —
   także gdy jakiegoś `device_type` czy maszyny jeszcze nie ma w danych.
2. Zrób zrzuty stron 0–10, przejdź skillem `design:design-critique`, nanieś poprawki.
3. Uaktualnij `dashboard/README.md` (nowe strony, Supabase jako jedyne źródło, uwaga o
   rosnącym zbiorze danych i maszynach wojtek / miłosz / maciej).
4. Wypisz krótko: które wykresy **potwierdzają** hipotezę, które ją **osłabiają lub jej
   przeczą**, i gdzie dane są za rzadkie, żeby przesądzić.
5. Dopilnuj, że `IMPLEMENTATION_STATUS.md` odzwierciedla stan końcowy — wszystko z „W
   trakcie" / „TODO" przeniesione do „Zrobione" albo świadomie zostawione z notatką
   dlaczego.
