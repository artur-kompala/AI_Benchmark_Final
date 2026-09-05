# Architektura

## Cel

System służy do zbierania porównywalnych danych o czasie wykonania, mocy chwilowej
i całkowitym zużyciu energii podczas treningu i wnioskowania modeli AI na trzech
klasach urządzeń: CPU, GPU (NVIDIA/AMD) i NPU (Intel Core Ultra), na potrzeby pracy
magisterskiej.

## Dwie niezależne warstwy

Świadomie **nie** budujemy jednej aplikacji z GUI, tylko dwa osobne procesy:

1. **`benchmark_runner`** — headless CLI. Jedyne zadanie: uruchomić eksperyment,
   zmierzyć go dokładnie i zapisać wynik. Brak GUI, brak serwera HTTP, brak
   zależności od dashboardu. Dzięki temu pomiar mocy nie jest zaburzony przez
   dodatkowe obciążenie (np. renderowanie wykresów w tle).
2. **`dashboard`** — aplikacja Streamlit tylko do odczytu z Supabase. Nigdy nie
   uruchamia benchmarków ani nie zapisuje do bazy.

Jedynym punktem styku jest **schemat bazy danych** (`db/migrations/`) — obie
warstwy niezależnie znają te same nazwy tabel/kolumn/wartości enum, ale nie
współdzielą importowanego kodu Pythona. To świadomy koszt (drobna duplikacja
stałych) na rzecz pełnej niezależności warstw i możliwości uruchamiania ich
na różnych maszynach bez wspólnego środowiska.

## Współdzielona baza danych (Supabase)

Wszystkie maszyny testowe (PC z GPU NVIDIA, PC z GPU AMD, laptop z Intel NPU)
zapisują wyniki do **tej samej** chmurowej bazy Supabase (Postgres). Dzięki temu
wyniki z różnych urządzeń agregują się automatycznie — nie trzeba ręcznie
przenosić plików między maszynami. Runner na każdej maszynie identyfikuje
"swoje" urządzenia przez `machine_name` + `device_type` (tabela `devices`),
więc wyniki z różnych komputerów nigdy się nie mieszają.

Schemat (pełny SQL w `db/migrations/0001_init_schema.sql`):

- `devices` — fizyczna maszyna + konkretne urządzenie obliczeniowe (jedna
  maszyna z CPU+GPU to dwa wiersze). `gpu_category` ('discrete'/'integrated',
  tylko dla `cuda`/`rocm`) rozróżnia GPU dyskretne od zintegrowanych (iGPU) —
  heurystyka po nazwie karty, patrz `benchmark_runner/devices/gpu_classification.py`.
- `runs` — jeden przebieg eksperymentu (jedna kombinacja
  task×model×device×batch_size×precision×phase). Każdy przebieg jest jednym
  z `repetitions` niezależnych powtórzeń tej samej kombinacji
  (`repetition_group_id`/`repetition_index`) — patrz "Powtórzenia pomiarów
  i statystyka" w `docs/measurement_methodology.md`.
- `power_samples` — surowe próbki mocy chwilowej (~100ms), wiele źródeł
  jednocześnie na jeden run (np. RAPL i codecarbon równolegle, do wzajemnej
  walidacji metod pomiaru programowego).
- `run_summary` — zagregowane metryki pochodne (energia, throughput,
  FLOPS/W, rozbieżność względem watomierza fizycznego, status kwantyzacji
  INT8) — relacja 1:1 z `runs`.
- `run_stats_summary` — mean/std/współczynnik zmienności energii i czasu,
  liczone po stronie runnera po zakończeniu wszystkich powtórzeń jednej
  kombinacji parametrów, jeden wiersz na `repetition_group_id` — patrz
  `docs/measurement_methodology.md`.

## Odporność na awarie

- **Brak internetu podczas serii pomiarowej**: `LocalBuffer` zapisuje rekordy
  do lokalnego pliku i retry'uje wysyłkę do Supabase po odzyskaniu połączenia
  — utrata sieci nie przerywa serii.
- **Niedostępne źródło pomiaru mocy** (np. RAPL niedostępny na danej maszynie,
  brak sterownika NPU): odpowiedni `PowerSampler.is_available()` zwraca
  `False`, sampler jest pomijany z logiem ostrzeżenia — benchmark **kontynuuje**
  i zapisuje `NULL`/adnotację zamiast się wywalić.
- **Błąd pojedynczego przebiegu** w serii wieloeksperymentowej: `runs.status`
  ustawiany na `'failed'` z `error_message`, pętla przechodzi do następnej
  konfiguracji zamiast przerywać całą serię.

## Pomiar mocy: wiele źródeł na raz

Dla jednego przebiegu może działać jednocześnie kilka samplerów mocy
(np. `rapl` + `codecarbon` na CPU), stąd `power_samples.source` jako kolumna
rozróżniająca pochodzenie zamiast jednej wartości per run — pozwala to na
wzajemną walidację metod pomiaru programowego w rozdziale metodologicznym
pracy (patrz `docs/measurement_methodology.md`).

Watomierz fizyczny (`source = 'smart_plug'`) jest traktowany jako niezależne
źródło referencyjne. `run_summary.power_discrepancy_pct` mierzy rozbieżność
między sumą energii z metod programowych a odczytem z gniazdka.

## Rozszerzalność

- **Nowe zadanie/model**: dodanie pliku w `tasks/<nazwa_zadania>/` z klasą
  dziedziczącą po `BaseTask` i dekoratorem `@register_task("nazwa")` —
  rdzeń (`orchestrator.py`) nie wymaga zmian.
- **Nowy backend watomierza** (np. Tasmota): implementacja `SmartPlugReader`
  w `power/smart_plug/`, wybór przez pole `power_meter.smart_plug.backend`
  w configu YAML — bez zmian w orchestratorze.
- **Nowe urządzenie/backend pomiaru mocy**: implementacja `BasePowerSampler`
  w `power/`, rejestrowana w `build_power_samplers()` zależnie od wykrytego
  urządzenia.

Zobacz pełny opis faz wdrożenia i status w głównym [`README.md`](../README.md).
