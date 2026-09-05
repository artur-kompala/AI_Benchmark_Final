# Konfiguracja laptopa z Intel Core Ultra (NPU)

> **Status**: ścieżka NPU jest zaimplementowana (Faza 3) — detekcja przez OpenVINO,
> alternatywna ścieżka przez ONNX Runtime/DirectML, eksport modeli PyTorch do ONNX,
> inferencja na `device="NPU"`. NPU wspiera w tym ekosystemie **tylko inferencję**
> (nie trening) i pomiar mocy działa jako fallback whole-system (patrz niżej).

## Wymagania sprzętowe/systemowe

- Procesor Intel Core Ultra (Meteor Lake lub nowszy) z wbudowanym NPU.
- Windows 11 (sterowniki Intel NPU wymagają aktualnego Windows 11).
- Zainstalowany **Intel NPU Driver** (Windows Update lub bezpośrednio ze strony
  Intela) — bez niego OpenVINO/DirectML nie zobaczą urządzenia `NPU`.

## Instalacja

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/requirements-common.txt
pip install -r requirements/requirements-npu.txt
pip install -e .
```

`torch`/`torchvision` (wariant CPU) są potrzebne do zbudowania modelu i eksportu do
ONNX (`tasks/image_classification/onnx_export.py`) — sama inferencja idzie przez
OpenVINO/ONNX Runtime, nie przez `torch` bezpośrednio.

## Weryfikacja detekcji NPU

```bash
python scripts/check_device_availability.py
```

Powinno pokazać `npu_openvino` (i ewentualnie `npu_directml`) na liście wykrytych
urządzeń. Jeśli NPU się nie pojawia, sprawdź ręcznie:

```python
from openvino import Core
print(Core().available_devices)  # powinno zawierac "NPU"
```

Jeśli `"NPU"` nie występuje — zwykle brakuje sterownika Intel NPU lub jest
nieaktualny.

## Uruchomienie eksperymentu na NPU

```bash
python -m benchmark_runner run --config configs/experiments/NPU/image_classification_mobilenet_cifar10.yaml
```

albo wymuszenie NPU na dowolnym innym configu (np. z folderu CPU/):

```bash
python -m benchmark_runner run --config configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml --device npu_openvino
```

### Ograniczenie: tylko inferencja

NPU w tym ekosystemie (OpenVINO / ONNX Runtime) **nie wspiera treningu** — jeśli
config zawiera fazę `train` dla urządzenia NPU, ten konkretny przebieg kończy się
czytelnym błędem (`runs.status = 'failed'`, `error_message` z wyjaśnieniem), a
reszta serii kontynuuje normalnie. Przykładowy config NPU
(`image_classification_mobilenet_cifar10_npu.yaml`) zawiera tylko `phases: [inference]`.

### Ograniczenie: precyzja

Na dzień pisania obsługiwana jest tylko `fp32` na ścieżce NPU — `int8`/`int4`
wymagają kwantyzacji modelu (NNCF dla OpenVINO / `onnxruntime.quantization` dla
DirectML), zaplanowanej w kolejnej iteracji. Żądanie `int8`/`int4` w configu
skutkuje ostrzeżeniem w logu i uruchomieniem z `fp32`.

## Pomiar mocy NPU

Nie ma obecnie publicznie udokumentowanego API zwracającego moc chwilową samego
NPU (w odróżnieniu od CPU/GPU, gdzie mamy RAPL/NVML/ROCm-SMI). Dlatego
`power/npu_power_meter.py` **zawsze** korzysta z fallbacku: pomiaru poboru mocy
całego systemu przez tempo rozładowania baterii (Windows Power API,
`CallNtPowerInformation`, `power/whole_system_meter.py`).

**Ten pomiar działa tylko gdy laptop jest odłączony od zasilacza i rozładowuje się
na baterii** — odłącz zasilacz przed uruchomieniem serii pomiarowej na NPU, inaczej
próbki mocy będą puste (`NULL`), a benchmark i tak będzie kontynuowany (zgodnie z
zasadą "brak danego źródła pomiaru nie przerywa benchmarku"), tylko bez danych o
mocy z tego źródła.

Każdy przebieg na NPU ma w tabeli `runs` pole `measurement_scope = "whole_system"`
— jawnie oznaczające, że zmierzono pobór mocy całej maszyny, a nie samego układu
NPU, do uwzględnienia przy interpretacji wyników w pracy (patrz
`docs/measurement_methodology.md`). Watomierz fizyczny (tryb manualny) pozostaje
niezależnym, dokładniejszym źródłem referencyjnym niezależnie od trybu zasilania.

## Uwaga o warm-up

Pierwsza kompilacja grafu OpenVINO (`Core.compile_model`) bywa jednorazowym
kosztem czasowym przy starcie — `warmup.iterations` w configu powinien być
wystarczająco duży, żeby wykluczyć ten efekt z właściwego pomiaru (przykładowy
config NPU używa `iterations: 5`).
