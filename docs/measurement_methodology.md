# Metodologia pomiaru energii i mocy

Ten dokument opisuje, *jak* system mierzy czas, moc i energię, oraz jakie
ograniczenia i założenia towarzyszą każdej metodzie — myślany jako materiał
źródłowy do rozdziału metodologicznego pracy magisterskiej.

## Czas wykonania

Mierzony przez `time.perf_counter()` (monotoniczny zegar wysokiej
rozdzielczości, odporny na zmiany zegara systemowego), od momentu
rozpoczęcia właściwego przebiegu (po warm-up) do jego zakończenia.
Warm-up (patrz niżej) jest **wyłączony** z pomiaru czasu i energii.

## Moc chwilowa — źródła programowe

Moc jest próbkowana w osobnym wątku co ok. 100 ms przez cały czas trwania
przebiegu (`power/sampler_thread.py`), niezależnie od głównego wątku
obliczeniowego, żeby nie zaburzać pomiaru czasu wykonania. Aktywne źródła
zależą od wykrytego sprzętu:

| Urządzenie | Źródła | Uwagi |
|---|---|---|
| CPU | `zenpower` (moduł jądra zenpower3, tylko AMD Zen), `rapl` (pyRAPL/pyJoules), `codecarbon` | Kolejność = priorytet w configu (`zenpower` > `rapl` > `codecarbon`), ale runner faktycznie wybiera źródło z największą liczbą zebranych próbek (patrz `_select_preferred_source`), nie sztywno pierwsze skonfigurowane. RAPL ma ograniczone wsparcie na Windows i bywa niekompletny/zablokowany uprawnieniami na AMD (patrz niżej). `zenpower` wymaga ręcznej instalacji, patrz `docs/setup_zenpower.md`. |
| GPU NVIDIA | `nvml` (pynvml, `nvmlDeviceGetPowerUsage`) | Odczyt bezpośrednio z karty przez sterownik NVIDIA. |
| GPU AMD (dyskretne) | `rocm_smi` (pyamdgpuinfo lub parsing `rocm-smi --showpower --json`) | Wymaga zainstalowanego stosu ROCm. |
| GPU AMD (zintegrowane, iGPU) | `amdgpu_igpu_native` (hwmon sterownika `amdgpu`) — fallback `amd_igpu_whole_system` | **Zawsze `whole_system`**, nie `device_only` — patrz sekcja "AMD iGPU" niżej i `docs/setup_amd_igpu.md`. |
| NPU (Intel) | `npu_native` — w praktyce zawsze fallback `whole_system` (patrz niżej) | Brak publicznego API natywnego odczytu mocy samego NPU. Patrz `docs/setup_npu_intel.md`. |
| GPU Intel (zintegrowane/dyskretne, przez OpenVINO) | `intel_gpu_native` — w praktyce zawsze fallback `whole_system` (jak NPU) | Brak publicznego API natywnego odczytu mocy samego GPU Intela. Patrz `devices/intel_gpu.py`. |
| NPU (AMD Ryzen AI / XDNA) | — brak integracji | Rozpoznane jako research spike i świadomie wyłączone z zakresu pracy — patrz `docs/research_amd_npu_xdna.md`. |

Gdy dane źródło jest niedostępne na danej maszynie, `PowerSampler.is_available()`
zwraca `False` — sampler jest pomijany (z logiem ostrzeżenia), pozostałe
źródła działają dalej, a w `power_samples` po prostu brak wpisów dla tego
źródła (zamiast przerywania całego pomiaru).

### Ograniczenie: RAPL na Windows

RAPL (Running Average Power Limit) to natywnie funkcja Intela dostępna
przez `/sys/class/powercap` na Linuksie. Na Windows dostęp do RAPL wymaga
dodatkowych sterowników/bibliotek i bywa niepewny w zależności od modelu
procesora i konfiguracji BIOS. Jeśli RAPL okaże się niedostępny na danej
maszynie testowej, `codecarbon` (który na Windows korzysta z innych
mechanizmów pomiaru, w tym estymacji na podstawie TDP gdy brak bezpośredniego
odczytu) pozostaje jedynym programowym źródłem dla CPU — należy to jawnie
odnotować przy interpretacji wyników danej maszyny.

### Sanity check: fizycznie nieprawdopodobne odczyty mocy CPU

**Empirycznie potwierdzone**: gdy RAPL jest zablokowany uprawnieniami (`PermissionError`
przy odczycie `/sys/class/powercap/intel-rapl/.../energy_uj`) lub po prostu przy krótkich
interwałach próbkowania, `codecarbon` potrafi wpaść w niestabilny wewnętrzny stan i zwrócić
pojedynczy odczyt mocy CPU rzędu setek watów — fizycznie niemożliwe dla laptopa (zaobserwowane
bezpośrednio na maszynie testowej tego projektu: 178.7 W i 244 W w pojedynczych odczytach,
przeplatane z prawidłowymi ~60–130 W). `RaplPowerSampler`/`CodecarbonPowerSampler`
(`power/rapl_meter.py`, `power/codecarbon_meter.py`) odrzucają teraz każdy **pojedynczy**
odczyt poza `MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE["cpu"]` (150 W) — traktowany jak nieudany
odczyt (`sample()` zwraca `None`), nie propagowany dalej. `core/orchestrator.py::
_check_power_plausibility` to druga linia obrony na poziomie **średniej** `avg_power_watts`
(np. gdyby kilka osobno "akceptowalnych" próbek dało nieprawdopodobną średnią) — w takim
przypadku pola mocy (`avg_power_watts`/`peak_power_watts`/`energy_joules_software`/...) są
zerowane, przebieg dostaje `status='partial'` z opisowym `error_message`, zamiast cicho
zapisać śmieciową wartość do Supabase. Próg dotyczy wyłącznie `device_type='cpu'` — GPU
dyskretne legalnie przekraczają 150 W.

### Fallback whole-system dla NPU i Intel GPU

Nie istnieje obecnie publicznie udokumentowane API zwracające moc chwilową
samego NPU (Intel Core Ultra) ani samego GPU Intela (zintegrowanego lub
dyskretnego, uruchamianego przez OpenVINO — patrz `devices/intel_gpu.py`), w
odróżnieniu od CPU (RAPL) i GPU NVIDIA/AMD (NVML/ROCm-SMI). Zarówno
`power/npu_power_meter.py`, jak i `power/intel_gpu_power_meter.py` używają więc
zawsze fallbacku `power/whole_system_meter.py`, który odczytuje tempo
rozładowania baterii — implementacja zależy od systemu operacyjnego:

- **Windows** — Windows Power API (`CallNtPowerInformation`, poziom
  `SystemBatteryState`, pole `Rate` w mW).
- **Linux** — sysfs `/sys/class/power_supply/BAT*/`: atrybut `power_now` (µW),
  gdy sterownik ACPI go udostępnia, w przeciwnym razie iloczyn `voltage_now`
  (µV) i `current_now` (µA).

**To działa wyłącznie gdy maszyna jest odłączona od zasilacza i rozładowuje się
na baterii** (Windows: `AcOnLine = False`, `Discharging = True`; Linux: atrybut
`status` baterii = `Discharging`) — żaden z systemów nie udostępnia
ogólnodostępnego API zwracającego pobór mocy całego systemu podczas pracy na
zasilaczu bez dedykowanego sprzętu/sterownika producenta płyty głównej.
Gdy maszyna jest podłączona do zasilania, próbki z tego źródła są puste (`None`)
— zgodnie z ogólną zasadą benchmarku, brak danego źródła pomiaru nie przerywa
przebiegu, tylko zostaje udokumentowany brakiem danych. Z tego powodu serie
pomiarowe na NPU i Intel GPU należy uruchamiać na baterii, jeśli zależy nam na
danych z tego źródła — watomierz fizyczny (Tuya) nie jest tu alternatywą, bo
mierzy pobór z gniazdka, a na baterii laptop celowo nie jest do niego podłączony.

## Watomierz fizyczny (źródło referencyjne)

Niezależnie od metod programowych, każdy przebieg może być zmierzony
inteligentnym gniazdkiem jako źródło referencyjne (`source = 'smart_plug'`
w `power_samples`).

**Stan obecny (Faza 4)**: dwa zaimplementowane backendy, wybierane przez
`power_meter.smart_plug.backend` w configu:

- **`tuya`** (domyslny na maszynach z watomierzem Tuya, np. ATORCH S1-B/W/T/H) -
  automatyczny, lokalny odczyt przez `tinytuya`, bez interakcji uzytkownika.
  Dwa niezalezne strumienie danych z tego samego urzadzenia:
  - ciagle probkowanie mocy chwilowej w tym samym interwale co inne zrodla
    (`source='smart_plug'` w `power_samples`, widoczne na wykresach obok
    RAPL/NVML/itd.),
  - odczyt **wewnetrznego licznika energii urzadzenia** na starcie i koncu
    przebiegu (`TuyaSmartPlugReader`) - gdy urzadzenie udostepnia taki DPS, to
    jest wartosc autorytatywna dla `run_summary.energy_joules_smart_plug`
    (odczyt ze sprzetowego akumulatora energii gniazdka, nie wyliczona z
    integracji probek mocy - bardziej wiarygodna niz re-integracja rzadko
    probkowanej krzywej mocy).

  **Fallback dla urzadzen bez licznika energii w DPS** (np. ATORCH S1BW - patrz
  `setup_smart_plug.md`): gdy `TuyaSmartPlugReader` nie zwroci odczytu (brak
  wlasciwego DPS), `run_summary.energy_joules_smart_plug` jest liczone przez
  calkowanie trapezoidalne ciaglych probek mocy z `source='smart_plug'` - tak
  samo jak `energy_joules_software` dla innych zrodel. Ta roznica metodologiczna
  (odczyt z akumulatora sprzetowego vs integracja probek) warto jawnie odnotowac
  przy interpretacji `power_discrepancy_pct` dla takich urzadzen w pracy.

  DPS (data points) protokolu Tuya roznia sie miedzy modelami urzadzen - patrz
  [`setup_smart_plug.md`](setup_smart_plug.md) po instrukcje weryfikacji
  wlasciwych DPS/skali na konkretnym egzemplarzu przed pomiarem referencyjnym.
  Brak `TUYA_DEVICE_ID`/`TUYA_LOCAL_KEY`/`TUYA_IP_ADDRESS` w `.env` (np. na
  maszynie bez podpietego watomierza) powoduje graceful fallback do trybu
  manualnego, z ostrzezeniem w logu.

- **`manual`** (fallback / maszyny bez watomierza Tuya): `ManualSmartPlugReader`
  - runner pauzuje na starcie i na koncu przebiegu, prosi o reczne wprowadzenie
    skumulowanego odczytu energii (Wh) z aplikacji gniazdka. Roznica odczytow
    koniec-start to zuzyta energia w trakcie przebiegu. Nie zapewnia ciaglych
    probek mocy (tylko `run_summary.energy_joules_smart_plug`).

Architektura (interfejs `SmartPlugReader`) przewiduje dalsze backendy (Tasmota,
Shelly - zarezerwowane nazwy w schemacie configu, niezaimplementowane) bez zmian
w rdzeniu runnera.

### Rozbieżność pomiaru: `power_discrepancy_pct`

Dla każdego przebiegu z aktywnym watomierzem liczona jest rozbieżność
procentowa między sumą energii z metod programowych
(`run_summary.energy_joules_software`) a odczytem z gniazdka
(`run_summary.energy_joules_smart_plug`):

```
power_discrepancy_pct = |energy_joules_software − energy_joules_smart_plug|
                         / energy_joules_smart_plug × 100
```

Jest to osobna metryka służąca do **oceny wiarygodności metod pomiaru
programowego** — nie jest korektą ani kalibracją, tylko udokumentowaną
rozbieżnością do dyskusji w pracy. Uwaga: gniazdko mierzy pobór mocy
**całej maszyny** (zasilacz, płyta główna, dyski, wentylatory — nie tylko
mierzone urządzenie obliczeniowe), więc pewna rozbieżność względem np.
samego odczytu NVML z karty graficznej jest oczekiwana i nie świadczy
sama w sobie o błędzie pomiaru programowego — to należy jasno rozróżnić
przy interpretacji wyników.

## `measurement_scope`

Każdy rekord w `runs` ma pole `measurement_scope`:

- `device_only` — moc zmierzona bezpośrednio z badanego urządzenia
  (CPU przez RAPL/codecarbon, GPU przez NVML/ROCm-SMI).
- `npu_only` — moc zmierzona natywnie z samego NPU. Nieużywane obecnie (brak
  publicznego API) — zarezerwowane na wypadek, gdyby takie API stało się
  dostępne w przyszłości.
- `whole_system` — fallback: zmierzono pobór mocy całego systemu przez tempo
  rozładowania baterii (patrz "Fallback whole-system dla NPU i Intel GPU" wyżej).
  **Zawsze** ustawiane dla przebiegów na NPU (`npu_openvino`/`npu_directml`) oraz
  na Intel GPU (`intel_gpu_openvino`) w obecnym stanie implementacji. **Wyniki
  z tym scope nie są bezpośrednio porównywalne** z wynikami `device_only` bez
  dodatkowego odjęcia szacowanego zużycia bazowego reszty systemu — do jawnego
  zaznaczenia w interpretacji wyników pracy.

## Energia — obliczenie z próbek mocy

`metrics/aggregation.py` liczy energię przez całkowanie trapezoidalne
próbek mocy po czasie:

```
E = Σ (P_i + P_{i+1}) / 2 × (t_{i+1} − t_i)     [J, gdy P w W i t w s]
```

Osobno per źródło (`rapl`, `codecarbon`, `nvml`, ...) — `run_summary.energy_joules_software`
przechowuje wartość z preferowanego/podstawowego źródła danego urządzenia
(np. `nvml` dla GPU NVIDIA), a wszystkie surowe próbki ze wszystkich źródeł
pozostają dostępne w `power_samples` do dalszej analizy porównawczej.

## Metryki pochodne

- **Energia/próbkę** (J/sample) = `energy_joules_software / samples_processed`.
- **Energia/epokę** (J/epoch) = `energy_joules_software / num_epochs` (tylko trening).
- **Throughput** (samples/s) = `samples_processed / duration_s`.
- **FLOPS/W** = `flops_per_sample / avg_power_watts` (FLOPS modelu szacowane
  statycznie przez `thop`/`fvcore`, niezależnie od pomiaru energii w danym
  przebiegu — to teoretyczna górna granica efektywności, nie zmierzona
  bezpośrednio). Liczone raz w `task.prepare()` z modelu fp32 PRZED ewentualną
  kwantyzacją (patrz `metrics/flops.py:estimate_flops` — dla modeli konwolucyjnych
  przez `torch.randn(1, *input_shape)`, dla DistilBERT przez parę tensorów
  `(input_ids, attention_mask)`, bo `forward()` nie przyjmuje pojedynczego
  tensora float) — `NULL`, gdy zadanie nie implementuje `estimate_flops_per_sample()`
  (obecnie: `llm_inference`) lub gdy oszacowanie się nie powiedzie (best-effort,
  nie przerywa przebiegu).
- **Accuracy** (`run_summary.accuracy`, 0–1) — dokładność klasyfikacji na zbiorze
  testowym, z `TaskStepResult.accuracy` (`image_classification`/`nlp_sentiment`).
  `NULL` dla `llm_inference` (brak tej metryki dla generowania tekstu). Kluczowe
  przy interpretacji `precision=int8`: energia niższa niż `fp32` nie jest sama
  w sobie argumentem za kwantyzacją, jeśli towarzyszy jej zauważalny spadek
  `accuracy` — porównuj obie kolumny razem, nie samą energię.

  **Accuracy referencyjna, nie "accuracy tej pętli wydajnościowej"** (`image_classification`):
  liczona RAZ na pełnym dostępnym zbiorze testowym (bez cyklicznego zawijania,
  `shuffle=False`) dla każdej unikalnej kombinacji (`model`, `precision`, `device`) i
  buforowana w pamięci procesu (`tasks/image_classification/task.py::_REFERENCE_ACCURACY_CACHE`)
  — **nie** liczona osobno per `batch_size`/`repetition`/`num_iterations`. Wcześniejsza
  implementacja liczyła accuracy z tej samej, zmiennej próbki co pomiar wydajności
  (sterowanej przez `batch_size × num_iterations` + cykliczne zawijanie dataloadera przy
  krótkim zbiorze testowym), przez co accuracy rosło monotonicznie wraz z
  `samples_processed` zamiast być stabilną właściwością wag — potwierdzone empirycznie
  (mobilenet: 0.32 przy 50 próbkach → 0.357 przy 3200 próbkach). Próg sanity-check
  (patrz niżej) sprawdza tę jedną, stabilną wartość referencyjną.

**Uwaga dla `llm_inference`**: kolumny `runs.throughput_samples_per_s` i
`run_summary.samples_processed`/`energy_per_sample_joules` reprezentują dla tego
zadania **tokeny wygenerowane**, nie próbki wejściowe — throughput LLM jest
konwencjonalnie wyrażany w tokenach/s, nie próbkach/s. Nazwy kolumn w schemacie SQL
pozostały generyczne (wspólne dla wszystkich zadań), więc przy analizie wyników
`llm_inference` w pracy interpretuj te wartości jako "na token", nie "na próbkę".

## Warm-up

Przed właściwym pomiarem wykonywana jest konfigurowalna liczba iteracji
rozgrzewających (`warmup.iterations` w configu YAML), nieobjęta pomiarem
czasu/energii. Cel: wykluczyć wpływ cold-startu frameworka (kompilacja
grafu, alokacja pamięci, JIT) oraz stanu przejściowego throttlingu
termicznego na wynik właściwego pomiaru.

### Adaptacyjne podniesienie `num_iterations` przy zbyt krótkich przebiegach

Moc jest próbkowana co `sampling_interval_ms` (domyślnie 100 ms) w osobnym wątku
(patrz "Moc chwilowa" wyżej). Przy bardzo krótkich przebiegach (typowo `batch_size=1`,
`duration_s` rzędu 0.1–0.2 s) sampler zdąży zebrać zaledwie 0–1 próbek w całym oknie
pomiaru, przez co `avg_power_watts` to praktycznie pojedynczy losowy odczyt, nie
rzeczywista średnia — potwierdzone empirycznie: 3 powtórzenia tej samej konfiguracji
dały 17.6 W / 7.3 W / 15.4 W (rozrzut 2–3×).

Mechanizm dwustopniowy — pierwsze oszacowanie okazało się niewystarczające samo w sobie:

1. **Wstępny strzał, PRZED pomiarem** — `core/orchestrator.py::_maybe_scale_up_num_iterations`
   szacuje czas trwania właściwego pomiaru na podstawie już zmierzonego czasu warm-up i
   podnosi `run_spec.inference.num_iterations` proporcjonalnie. **Empirycznie potwierdzone
   ograniczenie**: to oszacowanie systematycznie ZANIŻA rzeczywistą liczbę potrzebnych
   iteracji, bo pierwsze wywołanie po kompilacji OpenVINO/ONNX Runtime bywa
   nieproporcjonalnie wolniejsze od kolejnych (jednorazowy koszt "rozgrzania" cache
   kerneli silnika wykonawczego), a przy małej liczbie iteracji warm-up (domyślnie 5, w
   `--dry-run` tylko 1) ten pojedynczy wolny odczyt dominuje uśrednioną estymację —
   `per_iter_s` z warm-up wychodzi znacząco wyższe niż rzeczywisty koszt w stanie
   ustalonym. Skutek: po "przeskalowaniu" do np. 171 iteracji `power_samples_count` nadal
   wynosił 4–14, nie ~25 — bo prawdziwy czas był krótszy, niż estymacja zakładała.

2. **Autorytatywna korekta, retry loop PO każdej próbie** — `_run_single` liczy próbki
   mocy zebrane dotychczas (`PowerSamplingThread.sample_count()`, **bez zatrzymywania
   wątku** — kolejna próba kontynuuje TO SAMO okno pomiarowe, próbki się sumują, nie
   zaczynają od zera) i jeśli nadal poniżej progu, dolicza iteracje na podstawie
   **naprawdę zmierzonego czasu**. Maksymalnie 3 próby łącznie
   (`_MAX_POWER_SAMPLING_ATTEMPTS`) — po wyczerpaniu limitu przebieg zapisuje się z tym,
   co udało się zebrać (z ostrzeżeniem w logu), zamiast blokować serię w nieskończoność.
   `runs.duration_s`/`run_summary.samples_processed` sumują czas i liczbę próbek ze
   WSZYSTKICH prób tego przebiegu (spójne z `power_samples`, które też pochodzą z całego,
   nieprzerwanego okna) — `throughput_samples_per_s` pozostaje więc poprawnie zdefiniowane
   jako `samples_processed / duration_s` nawet przy wielu próbach.

   **Empirycznie potwierdzone drugie ograniczenie** (naprawione): pierwsza wersja korekty
   liczyła "ile iteracji potrzebowałaby JEDNA, IZOLOWANA próba, żeby SAMA zebrała próg
   próbek" z czasu WYŁĄCZNIE ostatniej próby — przy stałym czasie na iterację ta wartość
   jest stała niezależnie od numeru próby, więc po jednym kroku retry loop widział "brak
   postępu" i przerywał, MIMO ŻE realnie zebrana liczba próbek nadal wynosiła 0–1. Przyczyna:
   niektóre źródła (np. `codecarbon`) mają duży, JEDNORAZOWY narzut startowy — zmierzone
   empirycznie na maszynie testowej: `is_available()` (start `EmissionsTracker`) ~1.4 s +
   PIERWSZE wywołanie `sample()`/`flush()` ~3.1 s, zanim zacznie faktycznie próbkować w
   tempie `sampling_interval_ms`. Ten narzut pochłaniał większość czasu wczesnych prób,
   zanim jakiekolwiek próbki zdążyły napłynąć, a formuła — patrząc tylko na ostatnią próbę
   w izolacji — tego nie widziała. `_num_additional_iterations_needed()` liczy teraz
   DODATKOWE iteracje (nie nowy cel od zera) na podstawie SKUMULOWANEGO czasu wszystkich
   dotychczasowych prób: gdy już zebrano jakieś próbki, ekstrapoluje wprost z obserwowanej
   szybkości (`zebrane / skumulowany_czas`); gdy zebrano zero, podwaja dotychczasowy
   skumulowany czas jako kolejny krok (dając źródłu kolejną, większą szansę "dogonić"
   jednorazowy narzut startowy) — gwarantując monotoniczny wzrost zamiast plateau.

W obu krokach skalowanie jest ograniczone **czasowo** (`_MAX_POWER_SAMPLING_CUMULATIVE_
DURATION_S` = 20 s skumulowanego pomiaru na przebieg), nie wielokrotnością `num_iterations`
z configu. **Empirycznie potwierdzone trzecie ograniczenie** (naprawione): pierwsza wersja
limitu była wielokrotnością iteracji z configu (20×) — dla wysokiej przepustowości
(`batch_size=32/64`) nawet 20× mogło trwać krócej niż jednorazowy narzut startowy źródła
(np. 1000 iteracji przy `batch_size=64` zajmowało ~1.6–2.7 s, krócej niż ~1.4–3 s narzutu
codecarbon) — limit był więc osiągany, ZANIM sampler zdążył oddać pierwszą próbkę, zupełnie
niezależnie od tego, jak sprawnie działała reszta mechanizmu. Limit czasowy odzwierciedla
rzeczywiste ograniczenie (narzut startowy + docelowa liczba próbek), nie arbitralną
wielokrotność iteracji. Skalowanie nigdy nie zmniejsza `num_iterations` poniżej wartości z
configu. Finalna (ewentualnie wielokrotnie podniesiona) wartość trafia do zapisanego
`runs.num_iterations`, więc jest w pełni odtwarzalna z zapisanych danych.

**Fundamentalna poprawka: długożyjące źródła mocy współdzielone w całej serii.** Do tej
pory każdy POJEDYNCZY przebieg budował źródła mocy (np. `CodecarbonPowerSampler`) od zera,
płacąc pełny jednorazowy narzut startowy przy KAŻDYM przebiegu — potwierdzone empirycznie
na tej samej maszynie: `is_available()` (start `EmissionsTracker`) ~1.4 s + PIERWSZE
wywołanie `sample()`/`flush()` ~3.1 s, identycznie powtarzane w każdym z dziesiątek/setek
przebiegów jednej serii. `ExperimentRunner._get_persistent_sampling_thread` tworzy teraz
`PowerSamplingThread` (wraz z jego źródłami) RAZ na `device_type`, przy pierwszym użyciu w
całej serii (`run_from_config`), i reuzywa go dla wszystkich kolejnych przebiegów tego
samego typu urządzenia — `drain()` (nie `stop()`) na początku i końcu okna pomiarowego
każdego przebiegu czyści bufor bez zatrzymywania wątku, więc kolejne przebiegi korzystają
z już "rozgrzanego" źródła. Wątki są zamykane (`stop()`) dopiero na końcu całej serii.
Watomierz fizyczny (Tuya) pozostaje per-przebieg (osobny, krótkotrwały wątek) — jego
`start_run`/`end_run` z natury mierzy energię skumulowaną w oknie jednego `run_id`, więc
nie ma sensu go współdzielić między przebiegami. Zweryfikowane bezpośrednio na tej
maszynie: pierwszy przebieg z realnym `codecarbon` — 6.4 s na 19 próbek (płaci narzut);
drugi przebieg (reużyty wątek) — dokładnie 1.0 s na 10 próbek, pełne tempo
`sampling_interval_ms` od pierwszej próbki, zero powtórzonego narzutu.

**Wskaźnik wiarygodności**: `run_summary.power_samples_count` (liczba próbek mocy
faktycznie użytych do `avg_power_watts`/`peak_power_watts`, patrz
`metrics/aggregation.py::aggregate_run`) jest zapisywana dla KAŻDEGO przebiegu,
niezależnie od powyższego mechanizmu — dashboard ("Wiarygodność pomiaru") oznacza
przebiegi poniżej progu, żeby to było widoczne także dla danych sprzed tej poprawki.

**Diagnostyka źródła**: dla każdego przebiegu w logu (`core/orchestrator.py::
_finalize_and_persist`) pojawia się jawna informacja, KTÓRE źródło mocy (`preferred_source`,
np. `intel_gpu_native`, `rapl`, `nvml`) faktycznie dostarczyło próbki użyte do
`energy_joules_software`/`avg_power_watts`, ile próbek z tego źródła, i rozkład próbek
między wszystkimi aktywnymi źródłami — bez zgadywania z samej końcowej liczby, czy
np. `avg_power_watts=None` oznacza brak jakiegokolwiek źródła, czy tylko niewystarczającą
liczbę próbek z dostępnego źródła.

## Temperatura

Logowana jako kontekst (nie jako metryka pomiarowa sama w sobie) — pomaga
zinterpretować, czy dany przebieg mógł być dotknięty throttlingiem
termicznym. Zbierana best-effort (`power/temperature.py`); `None`, gdy
API danego urządzenia jej nie udostępnia.

## Powtórzenia pomiarów i statystyka

Pojedynczy pomiar czasu/energii jest podatny na szum (harmonogramowanie
systemu operacyjnego, throttling termiczny, tło procesów, wahania napięcia
zasilania) — jedna próbka nie pozwala odróżnić rzeczywistej różnicy między
urządzeniami/precyzjami od zwykłej wariancji pomiarowej. Dlatego każda
unikalna kombinacja parametrów (`model` × `device` × `precision` ×
`batch_size` × `phase`) jest wykonywana **wielokrotnie** — liczbę powtórzeń
ustala pole `repetitions` w configu YAML (domyślnie 3; niżej dla
kombinacji szczególnie kosztownych obliczeniowo, np. `repetitions: 2` dla
treningu ResNet-50 — patrz komentarz w
`configs/experiments/CPU/image_classification_resnet50_cifar10.yaml`).

**Wykonanie**: powtórzenia tej samej kombinacji są uruchamiane bezpośrednio
jedno po drugim (nie przeplatane z innymi kombinacjami), z **tym samym
seedem bazowym, ale unikalnym seedem per powtórzenie**
(`seed_powtorzenia = seed_bazowy + (repetition_index - 1)`, patrz
`config/schema.py:ExperimentConfig.expand_matrix`) — na początku każdego
przebiegu runner ustawia `random`/`numpy`/`torch` na ten seed
(`orchestrator._seed_everything`). Dzięki temu powtórzenia nie są
identyczne 1:1 (różne losowanie kolejności batchy, inicjalizacji wag przy
treningu itd.) — mierzymy więc rzeczywistą wariancję pomiaru, a nie tylko
determinizm powtórnego odtworzenia identycznych obliczeń.

**Zapis**: każde powtórzenie jest osobnym wierszem w `runs`, powiązanym
wspólnym `runs.repetition_group_id` (UUID wspólny dla wszystkich
powtórzeń danej kombinacji) i numerowanym przez `runs.repetition_index`
(od 1). Surowe dane każdego powtórzenia pozostają w pełni dostępne —
agregacja statystyczna niczego nie nadpisuje ani nie usuwa.

**Statystyka**: po zakończeniu wszystkich powtórzeń jednej grupy runner
liczy (w Pythonie, `orchestrator.ExperimentRunner._finalize_repetition_group`
— nie widok SQL, żeby logika agregacji żyła w jednym miejscu z resztą
pipeline'u metryk) średnią i **odchylenie standardowe próby** (`statistics.
stdev`, dzielnik `n-1`) czasu wykonania i energii z powtórzeń zakończonych
sukcesem (`status='completed'`), oraz współczynnik zmienności energii
(`coefficient_of_variation_energy = std_energy_joules / mean_energy_joules`
— znormalizowana miara rozrzutu, porównywalna między kombinacjami o różnej
skali energii, np. między CPU a NPU). Wynik trafia do tabeli
`run_stats_summary` (`db/migrations/0004_repetitions_stats_quantization.sql`),
jeden wiersz na `repetition_group_id`. Gdy tylko jedno powtórzenie zakończyło
się sukcesem, odchylenie standardowe jest zapisywane jako `0`, nie `NULL`
— odróżnia to "policzone na próbce wielkości 1" od "nie dało się policzyć".
Gdy żadne powtórzenie danej grupy się nie powiodło, wiersz w
`run_stats_summary` w ogóle nie powstaje (surowe `runs.status='failed'`
pozostają dostępne do analizy przyczyn).

**Ograniczenie**: jeśli zapis pojedynczego przebiegu trafił do lokalnego
bufora (`LocalBuffer`, brak sieci — patrz "Odporność na awarie" w
`docs/architecture.md`) i został wysłany do Supabase dopiero po
zakończeniu całej grupy powtórzeń, statystyka tej grupy została już
policzona bez niego (mniejsze `n_repetitions` niż skonfigurowane
`repetitions`) — świadomy kompromis, udokumentowany też jako komentarz
przy `ExperimentRunner._finalize_repetition_group` w kodzie.

## Kwantyzacja INT8

Poza `llm_inference` (ONNX Runtime, `onnxruntime.quantization` — patrz
`tasks/llm_inference/backends/onnxruntime_backend.py`), `int8` jest też
dostępne dla `image_classification` i `nlp_sentiment` na ścieżce PyTorch
(CPU/CUDA/ROCm), przez **dynamiczną kwantyzację PyTorch**
(`torch.quantization.quantize_dynamic`, patrz `utils/quantization.py`) —
podejście spójne z resztą projektu (bez dodatkowego kroku eksportu do ONNX
na tej ścieżce), w odróżnieniu od `llm_inference`, który i tak już
eksportuje do ONNX dla wszystkich precyzji.

**Tylko CPU**: kernele dynamicznej kwantyzacji PyTorch (FBGEMM/QNNPACK) są
CPU-only, dlatego `int8` jest w macierzy `precisions` **tylko** w plikach
configów z folderu `CPU/` (mirror ograniczenia `llm_inference_quantized.yaml`,
patrz `configs/experiments/README.md`) — config z `device: cuda`/`rocm` i
`int8` liczyłby się i tak na CPU, co błędnie etykietowałoby wynik w bazie.

**Tylko inferencja**: w tym projekcie nie trenuje się w INT8. Kombinacje
`precision=int8` + `phase=train` są automatycznie pomijane **przed** próbą
wykonania (`orchestrator._filter_unsupported_combos`), z czytelnym wpisem
w logu — nie próbą wykonania kończącą się błędem i wpisem
`runs.status='failed'`.

**Status kwantyzacji**: domyślny mapping dynamicznej kwantyzacji PyTorch
(`get_default_dynamic_quant_module_mappings()`) obejmuje `nn.Linear` (i
warstwy rekurencyjne: LSTM/GRU/RNN), ale **nie obejmuje `nn.Conv2d`** —
przekazanie `nn.Conv2d` do `quantize_dynamic()` nie jest błędem, te
warstwy po prostu zostają w fp32 bez ostrzeżenia z samego PyTorcha. Modele
oparte głównie o konwolucje (MobileNetV3, ResNet-50) są więc typowo
`'partial'` (skwantyzowana tylko końcowa warstwa `Linear`/klasyfikator), a
modele oparte głównie o `Linear` (DistilBERT — projekcje uwagi, FFN) są
typowo `'success'`. Ten status jest zapisywany per przebieg w
`run_summary.quantization_status` (`'success' | 'partial' | 'failed' |
NULL` dla precyzji innych niż `int8`) — **przy interpretacji wyników
`precision=int8` w pracy zawsze sprawdź to pole**: energia zmierzona dla
`'partial'` odzwierciedla model tylko **częściowo** skwantyzowany, nie
pełny INT8, i nie powinna być prezentowana jako "cały model w INT8" bez
tego zastrzeżenia.
