# benchmark_runner

Headless CLI runner benchmarkow CPU/GPU/NPU. Uruchamiany osobno na kazdej maszynie
testowej, wyniki trafiaja do wspoldzielonej bazy Supabase (patrz
[`../docs/setup_supabase.md`](../docs/setup_supabase.md)).

## Status implementacji (Faza 2 + 3 + 4 + 5 + 6)

Dziala pelny przeplyw dla zadania **klasyfikacji obrazow** (MobileNetV3/ResNet-50 na
CIFAR-10) na urzadzeniach **CPU / CUDA (NVIDIA) / ROCm (AMD)**: trening + inferencja,
pomiar czasu i mocy (RAPL/codecarbon/NVML/ROCm-SMI), zapis do Supabase z lokalnym
buforem na wypadek braku internetu.

**Watomierz fizyczny (Faza 4)**: automatyczny backend `tuya` (np. ATORCH S1-B/W/T/H)
przez lokalny protokol Tuya (`tinytuya`) - ciagle probkowanie mocy + odczyt licznika
energii urzadzenia, bez interakcji uzytkownika. Tryb `manual` pozostaje jako fallback
(dziala od Fazy 2) - patrz [`../docs/setup_smart_plug.md`](../docs/setup_smart_plug.md).

**NPU (Intel Core Ultra, Faza 3)**: dziala sciezka inferencji przez OpenVINO
(`device="NPU"`, podstawowa) i alternatywnie przez ONNX Runtime + DirectML - model
PyTorch jest eksportowany do ONNX i uruchamiany na NPU. NPU wspiera w tym
ekosystemie **tylko inferencje** (nie trening). Pomiar mocy dziala jako fallback
whole-system przez tempo rozladowania baterii (dziala tylko na baterii, nie na
zasilaczu) - patrz [`../docs/setup_npu_intel.md`](../docs/setup_npu_intel.md) i
[`../docs/measurement_methodology.md`](../docs/measurement_methodology.md).

**NLP i LLM (Faza 5)**: zadanie `nlp_sentiment` (fine-tuning + inferencja DistilBERT
na IMDB, CPU/CUDA/ROCm) i `llm_inference` (tylko inferencja, silnik ONNX Runtime z
kwantyzacja fp32/fp16/int8 w locie, lub llama-cpp-python z plikami GGUF dostarczonymi
przez uzytkownika) - patrz sekcja "Zadania NLP i LLM" nizej. Dashboard Streamlit -
patrz [`../dashboard/README.md`](../dashboard/README.md).

**Poprawki metodologiczne (Faza 6)**:
- Drugi model klasyfikacji obrazow **ResNet-50** (obok MobileNetV3), zaadaptowany pod
  CIFAR-10 (32x32) - zmniejszony stem konwolucyjny, patrz
  `tasks/image_classification/models.py:build_resnet50`. Osobny plik configu per
  urzadzenie (`image_classification_resnet50_cifar10.yaml`), celowo mniejsza liczba
  epok/powtorzen (wyzszy koszt obliczeniowy niz MobileNetV3).
- **Powtorzenia pomiarow** (`repetitions` w configu, domyslnie 3) - kazda kombinacja
  parametrow wykonywana wielokrotnie z roznym seedem, statystyka (mean/std/coefficient
  of variation) liczona automatycznie i zapisywana do `run_stats_summary` - patrz
  "Powtorzenia pomiarow i statystyka" w [`../docs/measurement_methodology.md`](../docs/measurement_methodology.md).
- **Kwantyzacja INT8** rozszerzona na `image_classification`/`nlp_sentiment` (dynamiczna
  kwantyzacja PyTorch, TYLKO CPU, TYLKO inferencja) - patrz "Kwantyzacja INT8" w tym
  samym dokumencie i `utils/quantization.py`.
- **Kategoria GPU** (dyskretne/zintegrowane) - heurystyka po nazwie karty, patrz
  `devices/gpu_classification.py`, uzywana przez dashboard.

**Rozszerzone wsparcie AMD (Faza 7)**:
- **`device_type="cpu"` na AMD Zen**: nowe zrodlo mocy `zenpower` (modul jadra
  `zenpower3`, realny pomiar telemetrii SVI2 z VRM) - opcjonalne, wymaga recznej
  instalacji, patrz [`../docs/setup_zenpower.md`](../docs/setup_zenpower.md). Kolejnosc
  domyslna `power_meter.cpu_sources`: `zenpower` > `rapl` > `codecarbon`. Bez
  zainstalowanego modulu runner dziala jak dotychczas (fallback na `rapl`/`codecarbon`).
- **Nowy `device_type="amd_igpu"`** (zintegrowane GPU AMD, np. Vega w Ryzen 5000H,
  Radeon 780M w Ryzen 7040/8040) - pomiar mocy przez `amdgpu_igpu_native` (hwmon
  sterownika `amdgpu`, dziala bez ROCm) zweryfikowany na sprzecie; sciezka compute
  (PyTorch+ROCm, osobny venv) **zweryfikowana dla fazy inference** na "Oli" (Vega/gfx90c,
  wymaga `HSA_OVERRIDE_GFX_VERSION=9.0.0` - rocBLAS w ROCm 6.2 nie ma natywnej biblioteki
  Tensile dla gfx90c) - faza `train` pozostaje niezweryfikowana. `measurement_scope`
  zawsze `whole_system` (odczyt to `PPT` calego SoC, nie samego GPU). Configi w
  `configs/experiments/GPU_AMD_IGPU/` (mobilenet, resnet50, nlp_distilbert), patrz
  [`../docs/setup_amd_igpu.md`](../docs/setup_amd_igpu.md).
- **AMD NPU (Ryzen AI / XDNA)**: rozpoznane jako research spike, **swiadomie poza
  zakresem** (ekosystem software'owy - oficjalny Vitis AI EP w early access, alternatywa
  community wymaga budowy wlasnego toolchaina) - raport i uzasadnienie w
  [`../docs/research_amd_npu_xdna.md`](../docs/research_amd_npu_xdna.md). Zadne pliki w
  `configs/experiments/NPU/` ani kod orchestratora nie zostaly zmienione w ramach tego
  rozpoznania.

## Instalacja

### 1. Srodowisko wirtualne

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 2. PyTorch (osobno, zalezne od maszyny)

**PC z GPU NVIDIA:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**PC z GPU AMD:** zobacz [`../docs/setup_windows_gpu.md`](../docs/setup_windows_gpu.md) -
oficjalne wsparcie torch+ROCm na Windows jest ograniczone, moze wymagac WSL2.

**Sama maszyna CPU / laptop NPU (do czasu Fazy 3):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Reszta zaleznosci

```bash
pip install -r requirements/requirements-common.txt
pip install -r requirements/requirements-nvidia.txt   # tylko na PC z NVIDIA
pip install -r requirements/requirements-amd.txt       # tylko na PC z AMD
pip install -r requirements/requirements-npu.txt        # tylko na laptopie z NPU (patrz uwaga o kolejnosci instalacji w pliku)
pip install -r requirements/requirements-dev.txt        # do uruchamiania testow
```

`requirements-common.txt` zawiera tez zaleznosci zadania `llm_inference`
(`onnxruntime`, `onnxconverter-common`, `llama-cpp-python`) - `llama-cpp-python`
kompiluje sie z zrodel przy instalacji i moze potrwac kilka minut (nie wymaga
dodatkowych narzedzi na typowym Windows+Python 3.11, ale jesli instalacja zawiedzie,
reszta CLI dziala normalnie - silnik `llama_cpp` zglosi czytelny blad importu dopiero
przy faktycznej probie uzycia).

### 4. Instalacja pakietu w trybie edytowalnym

Pozwala uruchamiac `python -m benchmark_runner ...` z dowolnego katalogu i importowac
pakiet w testach bez recznego ustawiania `PYTHONPATH`:

```bash
pip install -e .
```

### 5. Dane logowania Supabase

```bash
copy .env.example .env
```

Uzupelnij `SUPABASE_URL` i `SUPABASE_KEY` (patrz [`../docs/setup_supabase.md`](../docs/setup_supabase.md)).

## Weryfikacja instalacji

```bash
python scripts/check_device_availability.py
```

Powinno wypisac wykryte urzadzenie (co najmniej `cpu`) oraz dostepne zrodla pomiaru
mocy na tej maszynie (np. `codecarbon: DOSTEPNE`, `rapl: niedostepne` - typowe na
Windows, patrz [`../docs/measurement_methodology.md`](../docs/measurement_methodology.md)).

## Uruchomienie pierwszego eksperymentu

Configi sa pogrupowane w foldery per typ sprzetu - wybierz folder odpowiadajacy
Twojej maszynie, `device` jest juz ustawiony na sztywno w kazdym pliku, wiec **nie
trzeba edytowac YAML ani pamietac `--device`** przy przechodzeniu miedzy maszynami
(patrz [`configs/experiments/README.md`](configs/experiments/README.md)):

```bash
python -m benchmark_runner run --config configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml
```

Jesli mimo wszystko chcesz wymusic inne urzadzenie niz to w configu, flaga `--device`
ma pierwszenstwo:

```bash
python -m benchmark_runner run --config configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml --device cuda
```

Jesli w configu `power_meter.smart_plug.backend: "tuya"` i masz skonfigurowany
watomierz w `.env`, pomiar jest w pelni automatyczny. Bez tego (albo przy
`backend: "manual"`) runner zatrzyma sie na poczatku i na koncu kazdego przebiegu
i poprosi o reczne wpisanie skumulowanej energii (Wh) z aplikacji gniazdka - patrz
[`../docs/setup_smart_plug.md`](../docs/setup_smart_plug.md).

### Uruchomienie na GPU NVIDIA / AMD / NPU

```bash
python -m benchmark_runner run --config configs/experiments/GPU_NVIDIA/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_AMD/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/NPU/image_classification_mobilenet_cifar10.yaml
```

NPU wspiera tylko inferencje (nie trening), a wiarygodny pomiar mocy programowej
przez fallback whole-system wymaga pracy na baterii (bez zasilacza) - szczegoly w
[`../docs/setup_npu_intel.md`](../docs/setup_npu_intel.md).

### Zadania NLP i LLM (Faza 5)

**NLP (`nlp_sentiment`)** - fine-tuning + inferencja DistilBERT na IMDB, dziala tak
samo jak `image_classification` (CPU/CUDA/ROCm, brak jeszcze sciezki NPU dla tego
zadania):

```bash
python -m benchmark_runner run --config configs/experiments/CPU/nlp_distilbert_imdb.yaml
```

**LLM (`llm_inference`)** - tylko inferencja (LLM nie sa tu trenowane od zera).
Domyslny silnik `onnxruntime` eksportuje maly model z Hugging Face do ONNX i
kwantyzuje w locie (fp32/fp16/int8) - dziala od razu, bez dodatkowych plikow.
Dostepny tylko w folderze `CPU/` (backend dziala obecnie na CPU execution provider
niezaleznie od wybranego urzadzenia, patrz `configs/experiments/GPU_NVIDIA/README.md`):

```bash
python -m benchmark_runner run --config configs/experiments/CPU/llm_inference_quantized.yaml
```

Alternatywny silnik `llama_cpp` wymaga samodzielnie dostarczonego pliku `.gguf`
(np. z Hugging Face Hub) - ustaw w configu:

```yaml
llm:
  engine: "llama_cpp"
  gguf_model_path: "C:/sciezka/do/model.Q4_K_M.gguf"
```

Porownanie precyzji dla `llama_cpp` realizuje sie przez wskazanie **innego pliku
GGUF per przebieg** (kwantyzacja jest juz "wpieczona" w plik), w przeciwienstwie do
`onnxruntime`, gdzie konwersja/kwantyzacja dzieje sie automatycznie w locie.

### Szybki smoke test (`--dry-run`)

Uruchamia 1 epoke treningu i 3 iteracje inferencji zamiast pelnej konfiguracji z
YAML - przydatne do sprawdzenia, ze caly przeplyw (urzadzenie -> pomiar mocy ->
zapis w Supabase) dziala bez wyjatkow, zanim odpali sie pelna, dlugotrwala serie
pomiarowa. Wynik nadal trafia do Supabase (mozna go zweryfikowac w Table Editor).

```bash
python -m benchmark_runner run --config configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml --dry-run
```

Pierwsze uruchomienie pobierze automatycznie zbior CIFAR-10 (~170 MB) do
`benchmark_runner/data/cifar10/` (poza repo, w `.gitignore`).

## Testy jednostkowe

```bash
pip install -r requirements/requirements-dev.txt
pytest
```

Testy nie wymagaja GPU ani prawdziwego polaczenia z Supabase (klient jest
mockowany) - dzialaja identycznie na kazdej maszynie.

## Uzycie na wielu maszynach

Ten sam kod (ten katalog) dziala identycznie na kazdej maszynie testowej - roznica
jest tylko w tym, ktore `requirements-*.txt` i jaki wariant instalacji PyTorch
zostaly uzyte. Detekcja urzadzenia jest automatyczna
(`benchmark_runner.devices.detection`), z mozliwoscia nadpisania przez `--device`.
Wszystkie maszyny zapisuja do tej samej, wspoldzielonej bazy Supabase - wyniki z
PC (NVIDIA/AMD) i laptopa (NPU, od Fazy 3) agreguja sie automatycznie.

## Struktura konfiguracji eksperymentu

Przyklad: [`configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml`](configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml).
Metoda `expand_matrix()` (w `config/schema.py`) rozwija listy `models`/`devices`/
`precisions`/`batch_sizes`/`phases` w kartezjanski iloczyn, kazda kombinacje
powielajac dodatkowo `repetitions` razy (domyslnie 3, patrz pole `repetitions` w
configu) - jeden plik configu moze zdefiniowac wiele przebiegow (runs) wykonywanych
sekwencyjnie, powtorzenia tej samej kombinacji jedno po drugim. Blad pojedynczego
przebiegu jest logowany i **nie przerywa** reszty serii. Kombinacje `precision=int8`
+ `phase=train` sa pomijane automatycznie przed wykonaniem (w tym projekcie nie
trenuje sie w INT8) - patrz `core/orchestrator.py:_filter_unsupported_combos`.

## Rozszerzanie o nowe zadanie

1. Utworz katalog `src/benchmark_runner/tasks/<nazwa_zadania>/` z klasa dziedziczaca
   po `BaseTask` (`core/base_task.py`) i dekoratorem `@register_task("nazwa")`
   (wzor: `tasks/image_classification/task.py`).
2. Zaimportuj nowy modul zadania w `core/orchestrator.py` (obok istniejacego importu
   `image_classification`), zeby dekorator sie wykonal i zarejestrowal zadanie.
3. Dodaj plik configu w odpowiednich folderach `configs/experiments/<CPU|GPU_NVIDIA|GPU_AMD|GPU_AMD_IGPU|GPU_INTEL|NPU>/`
   z `task: "<nazwa_zadania>"` (pomin foldery urzadzen, ktorych nowe zadanie jeszcze
   nie wspiera - patrz `configs/experiments/README.md`).

Rdzen runnera (`orchestrator.py`) nie wymaga zadnych innych zmian.
