# Konfiguracja maszyny ze zintegrowanym GPU AMD (iGPU)

> **Status**: **zweryfikowane (faza inference)**. Pomiar mocy (`power_meter.amd_igpu_sources`)
> i faza `inference` dla `image_classification`/`mobilenet_v3` są potwierdzone jako
> działające na maszynie testowej "Ola" (Ryzen 7 5800HS, iGPU Radeon Vega/Cezanne,
> architektura `gfx90c`) — patrz "Część 2: Compute" niżej za konkretną komendą i wynikiem.
> **Faza `train` (fine-tuning) i pozostałe modele/zadania (ResNet-50, DistilBERT) NIE
> zostały jeszcze zweryfikowane** — configi w `configs/experiments/GPU_AMD_IGPU/` mają
> celowo `phases: ["inference"]`. Przeczytaj ten dokument w całości przed próbą
> rozszerzenia zakresu.

## Dwie niezależne części tej ścieżki

1. **Pomiar mocy** (`power/amdgpu_igpu_power_meter.py`) — czyta moc bezpośrednio ze
   sterownika jądra `amdgpu` przez `hwmon` w sysfs. Działa **niezależnie** od ROCm/PyTorch
   — to zwykły sterownik jądra Linuksa, obecny na każdej maszynie z GPU AMD (dyskretnym
   lub zintegrowanym) obsługiwanym przez `amdgpu`, bez dodatkowej instalacji.
2. **Compute** (`devices/amd_igpu.py`, backend `pytorch`) — wymaga PyTorch skompilowanego
   z ROCm i żeby ROCm w ogóle rozpoznawał dany iGPU jako urządzenie obliczeniowe. To jest
   część niezweryfikowana.

Możesz mieć działający pomiar mocy BEZ działającego compute — `python scripts/
check_device_availability.py` pokaże `amdgpu_igpu_native: DOSTEPNE` nawet gdy
`amd_igpu` w ogóle nie pojawi się na liście "Wykryte urządzenia obliczeniowe".

## Część 1: Pomiar mocy (zweryfikowane)

Nic do zainstalowania — jeśli masz standardowy sterownik `amdgpu` (domyślny w
nowoczesnych jądrach Linuksa dla GPU AMD, dyskretnych i zintegrowanych), pomiar mocy
działa od razu.

### Weryfikacja

```bash
python scripts/check_device_availability.py
```

Powinno pokazać `amdgpu_igpu_native (AMD GPU zintegrowane): DOSTEPNE`. Ręcznie:

```bash
find /sys/class/drm/card*/device/hwmon -name name -exec sh -c 'echo {}: $(cat {})' \; | grep amdgpu
cat /sys/class/drm/cardX/device/hwmon/hwmonY/power1_average   # X, Y ze sciezki wyzej
cat /sys/class/drm/cardX/device/hwmon/hwmonY/power1_label     # zwykle "PPT"
```

### Ważne zastrzeżenie metodologiczne: `PPT` to pobór CAŁEGO SoC, nie samego GPU

Zmierzone bezpośrednio na maszynie testowej "Ola": `power1_input` = 15 000 000 (15.0 W),
`power1_label` = `PPT` (**P**ackage **P**ower **T**racking). To jest z definicji AMD pobór
mocy **całego pakietu SoC** (CPU + iGPU razem), **nie** samego bloku graficznego. Dla
porównania, jednoczesny odczyt whole-system (tempo rozładowania baterii) dał 25.9 W —
bardzo bliska wartość, spójna z tezą że `PPT` to w praktyce "prawie cały system".

**Dlatego `measurement_scope` dla `device_type='amd_igpu'` jest zawsze `'whole_system'`**,
niezależnie od tego, czy dane pochodzą z `amdgpu_igpu_native` (hwmon) czy z fallbacku
`amd_igpu_whole_system` (bateria) — **nie interpretuj** wyników z tej ścieżki jako
"pobór mocy samego iGPU" przy porównaniach z `device_only` (CPU/CUDA/ROCm dyskretne w tym
projekcie). Patrz też `docs/measurement_methodology.md`.

Fallback whole-system (`amd_igpu_whole_system`, tempo rozładowania baterii) działa
**tylko na baterii** — jak dla NPU/Intel GPU, patrz `docs/measurement_methodology.md`.

## Część 2: Compute — inference zweryfikowane, reszta nie

### Krytyczne ograniczenie: osobny venv, niekompatybilny z GPU NVIDIA na tej samej maszynie

Jeśli Twoja maszyna ma **jednocześnie** dyskretne GPU NVIDIA (jak "Ola" — RTX 3050 Ti) i
zintegrowane GPU AMD, **nie da się** mieć jednego środowiska Python z PyTorch
obsługującym oba naraz — build `torch+cu124` (CUDA, dla NVIDIA) i build `torch+rocm` (dla
AMD) to różne, wzajemnie wykluczające się pakiety. Zweryfikowane bezpośrednio: w
istniejącym venv projektu (z `torch==2.6.0+cu124` dla `../GPU_NVIDIA/`)
`torch.version.hip` to `None` — `devices/amd_igpu.py` poprawnie zwraca `is_available()
== False`, ale to NIE oznacza że iGPU jest niedostępny sprzętowo, tylko że TEN venv go nie
widzi.

Żeby przetestować ścieżkę `amd_igpu`, potrzebujesz **osobnego** venv:

```bash
python3.11 -m venv venv-rocm
source venv-rocm/bin/activate
# Sprawdz aktualna komende na pytorch.org (wybierz ROCm jako compute platform) -
# przykladowo dla ROCm 6.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements/requirements-common.txt -r requirements/requirements-amd.txt
pip install -e .
```

### Możliwa potrzeba `HSA_OVERRIDE_GFX_VERSION`

ROCm oficjalnie wspiera ograniczoną listę architektur GPU. Wiele APU (w tym Vega w
Ryzen 5000H-series — architektura `gfx90c`, i Radeon 780M w Ryzen 7040/8040-series —
`gfx1103`) nie jest oficjalnie wspieranych, ale często działa z workaroundem
`HSA_OVERRIDE_GFX_VERSION`, który każe ROCm traktować iGPU jako architekturę zbliżoną,
oficjalnie wspieraną:

```bash
# Przyklad dla Vega/gfx90c (Ryzen 5000H) - traktuj jako gfx902:
export HSA_OVERRIDE_GFX_VERSION=9.0.0
# Przyklad dla 780M/gfx1103 (Ryzen 7040/8040) - traktuj jako gfx1100:
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Dokładna wartość zależy od konkretnej architektury i wersji ROCm — sprawdź aktualne
źródła community (np. ROCm GitHub issues) dla Twojego konkretnego APU, to się zmienia
między wydaniami ROCm.

### Potwierdzone na żywo: Vega/gfx90c + ROCm 6.2 + `HSA_OVERRIDE_GFX_VERSION=9.0.0`

Na maszynie testowej "Ola" próba uruchomienia bez override kończyła się błędem rocBLAS:

```
rocBLAS error: Cannot read .../torch/lib/rocblas/library/TensileLibrary.dat: Illegal seek
for GPU arch : gfx90c
List of available TensileLibrary Files: ... gfx900, gfx906, gfx908, gfx90a, gfx942,
gfx1030, gfx1100 (BRAK gfx90c)
```

Ten konkretny build `torch==2.5.1+rocm6.2` (instalowany przez `pip install torch
torchvision --index-url https://download.pytorch.org/whl/rocm6.2`) nie ma skompilowanej
biblioteki Tensile dla `gfx90c` — tylko dla architektur z listy wyżej. `gfx900` jest
najbliższym krewnym Vegi na tej liście, więc:

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

**zadziałało bez błędów** — pełna seria 12 przebiegów (`image_classification_mobilenet_
cifar10.yaml`, faza `inference`, batch_size 1/4/32/64 × 3 powtórzenia) zakończyła się
poprawnie, wyniki zapisane do Supabase. Po drodze widoczne dwa łagodne, nieszkodliwe
ostrzeżenia (nie błędy): `FutureWarning` o `torch.load(weights_only=False)` (niezwiązane z
AMD — dotyczy sposobu wczytywania checkpointu w całym projekcie) i `UserWarning:
Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to
hipblas` (PyTorch samo przełącza się na kompatybilny backend BLAS — nieszkodliwe).

Zmierzona moc (`amdgpu_igpu_native`, PPT całego SoC) rosła sensownie z batch_size: mediana
~17.5 W przy batch_size=1 do ~31.5 W przy batch_size=64 — spójne z oczekiwanym wzorcem
(większe obciążenie GPU/CPU przy większych batchach).

Jeśli Twoja architektura APU jest inna niż `gfx90c` (np. `gfx1103` dla Radeon 780M), ten
konkretny wynik NIE przenosi się automatycznie — potrzebujesz własnej weryfikacji z
odpowiednią wartością `HSA_OVERRIDE_GFX_VERSION` (patrz przykład dla `gfx1103` wyżej).

### Weryfikacja compute

```bash
source venv-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=...  # jesli potrzebne, patrz wyzej
python scripts/check_device_availability.py
```

Szukaj `amd_igpu: <nazwa karty> (backend=pytorch, ...)` w sekcji "Wykryte urządzenia
obliczeniowe". Jeśli się nie pojawia mimo `HSA_OVERRIDE_GFX_VERSION`, ROCm prawdopodobnie
faktycznie nie wspiera tej architektury w zainstalowanej wersji — sprawdź `rocminfo` czy
w ogóle wykrywa GPU.

### Jeśli compute zadziała

Wszystkie trzy configi w `configs/experiments/GPU_AMD_IGPU/` (`image_classification_
mobilenet_cifar10.yaml` — zweryfikowany na żywo, patrz wyżej; `image_classification_
resnet50_cifar10.yaml`; `nlp_distilbert_imdb.yaml` — te dwa jeszcze nie przetestowane
osobno) są gotowe do użycia — celowo ograniczone do `phases: ["inference"]` (nie
`train`), bo dokładność/stabilność treningu na tej ścieżce jest dodatkowym nieznanym.
Warianty `image_classification` wymagają tego samego checkpointu referencyjnego co
CPU/GPU_NVIDIA/GPU_AMD/GPU_INTEL (patrz `docs/setup_intel_gpu.md`, sekcja o
checkpointach) — skopiuj `data/checkpoints/mobilenet_v3_cifar10_seed42.pt` na tę
maszynę.

### Jeśli compute NIE zadziała

To nie jest porażka konfiguracji — udokumentuj to jako świadome ograniczenie
metodologiczne (analogicznie do NPU AMD/XDNA, patrz
`docs/research_amd_npu_xdna.md`) i użyj pomiaru mocy iGPU (Część 1 wyżej) niezależnie,
jeśli jest to przydatne samo w sobie (np. do scharakteryzowania poboru mocy SoC w
spoczynku/pod obciążeniem innym zadaniem).

## Rozróżnienie dyskretne vs zintegrowane GPU AMD

`devices/amd_igpu.py` i `devices/rocm.py` (dyskretne karty) rozróżniają urządzenia po
nazwie zwracanej przez `torch.cuda.get_device_name()`, tą samą heurystyką co
`devices/gpu_classification.py` (patrz `_AMD_INTEGRATED_PATTERNS`) — `amd_igpu.py` bierze
TYLKO urządzenia sklasyfikowane jako `'integrated'`, `rocm.py` (bez zmian w tym zadaniu)
nadal zawsze bierze indeks 0 niezależnie od kategorii. Na wszystkich znanych dziś
maszynach testowych to nigdy nie koliduje — żadna maszyna z dyskretną kartą AMD (Artur PC)
nie ma iGPU (desktop), a żadna maszyna z iGPU AMD (Ola, Proksza) nie ma dyskretnej karty
AMD (ich dyskretne GPU to NVIDIA, obsługiwane osobno). Gdyby kiedyś pojawiła się maszyna
testowa z OBIEMA jednocześnie, `devices/rocm.py` wymagałby analogicznej poprawki (wybór
pierwszego urządzenia sklasyfikowanego jako `'discrete'`, nie indeksu 0) — celowo nie
wprowadzonej teraz, żeby nie zmieniać istniejącego, przetestowanego zachowania na
znanym sprzęcie (Artur PC / RX 7800 XT) bez potrzeby.
