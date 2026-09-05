# Konfiguracja maszyny z Intel GPU (zintegrowane Iris Xe/Arc lub dyskretne Arc)

> **Status**: ścieżka Intel GPU (Faza 6) jest zaimplementowana tą samą metodą co NPU
> (patrz [`setup_npu_intel.md`](setup_npu_intel.md)) — detekcja i inferencja przez
> OpenVINO GPU plugin (`device_name="GPU"`), eksport modeli PyTorch do ONNX. Wspiera
> **tylko inferencję** (nie trening) i pomiar mocy działa jako fallback whole-system
> (patrz niżej) — identycznie jak NPU. Jeśli już skonfigurowałeś NPU na tej maszynie,
> większość poniższego już masz zainstalowane.

## Wymagania sprzętowe/systemowe

- Dowolne GPU Intela widoczne przez sterownik grafiki: zintegrowane (Iris Xe, Arc w
  Core Ultra/13.-14. generacji) lub dyskretne (Arc A-/B-series).
- Windows 11 lub Linux (obie wspierane, także pod kątem pomiaru mocy — patrz
  niżej) z aktualnym **sterownikiem grafiki Intel** — bez niego OpenVINO nie
  zobaczy urządzenia `GPU`.

## Instalacja

Identyczna jak dla NPU:

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/requirements-common.txt
pip install -r requirements/requirements-npu.txt
pip install -e .
```

## Weryfikacja detekcji

```bash
python scripts/check_device_availability.py
```

Powinno pokazać `intel_gpu_openvino` na liście wykrytych urządzeń, z etykietą typu
`Intel(R) Arc(TM) Graphics (iGPU)` lub `Intel(R) Iris(R) Xe Graphics`. Jeśli GPU się
nie pojawia, sprawdź ręcznie:

```python
from openvino import Core
print(Core().available_devices)  # powinno zawierac "GPU"
```

## Kategoria dyskretne vs zintegrowane — wykrywana automatycznie

W odróżnieniu od GPU NVIDIA/AMD (gdzie kategoria jest heurystyką po nazwie karty,
patrz `devices/gpu_classification.py`), dla Intel GPU kategoria jest odczytywana
**wprost ze sterownika** przez natywną właściwość OpenVINO `DEVICE_TYPE`
(`devices/intel_gpu.py`) — nie wymaga żadnej konfiguracji ani ręcznej korekty.

## Uruchomienie eksperymentu

```bash
python -m benchmark_runner run --config configs/experiments/GPU_INTEL/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_INTEL/image_classification_resnet50_cifar10.yaml
```

### Ograniczenie: tylko inferencja, tylko fp32

Dokładnie te same ograniczenia co dla NPU (patrz `setup_npu_intel.md`) — brak
treningu na tej ścieżce, `int8`/`int4` wymagają kwantyzacji NNCF (planowane w
kolejnej iteracji, na razie skutkuje ostrzeżeniem w logu i uruchomieniem z `fp32`).

## Pomiar mocy — tylko na baterii

Tak jak NPU: brak publicznego API natywnego odczytu mocy samego GPU Intela, więc
`power/intel_gpu_power_meter.py` **zawsze** korzysta z fallbacku whole-system (tempo
rozładowania baterii) — na Windows przez Windows Power API, na Linuksie przez
sysfs `/sys/class/power_supply/BAT*/` (patrz `docs/measurement_methodology.md`).
**Odłącz zasilacz przed uruchomieniem serii pomiarowej**, inaczej próbki mocy z
tego źródła będą puste (benchmark i tak będzie kontynuowany). Watomierz fizyczny
na laptopie z tej ścieżki **nie ma sensu jako alternatywa** — mierzy pobór z
gniazdka, a na baterii laptop jest celowo od niego odłączony; traktuj whole-system
jako jedyne źródło referencyjne dla serii na baterii.

## Uwaga o warm-up i pierwszej kompilacji

Pierwsza kompilacja grafu OpenVINO na GPU (`Core.compile_model(device_name="GPU")`)
bywa zauważalnie wolniejsza niż kolejne (kompilacja/cache kerneli OpenCL) —
`warmup.iterations` w configu wyklucza ten efekt z właściwego pomiaru. Jeśli
pierwsze uruchomienie (nawet `--dry-run`) wydaje się "zawieszone" na kilka minut,
to zwykle właśnie ta jednorazowa kompilacja, nie błąd.
