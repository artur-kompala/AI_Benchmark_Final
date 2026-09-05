# Configi dla laptopa ze zintegrowanym GPU AMD (iGPU/APU)

Wszystkie configi w tym folderze maja `devices: [amd_igpu]` ustawione na sztywno oraz
`power_meter.amd_igpu_sources: [amdgpu_igpu_native, amd_igpu_whole_system]` -
uruchamiaj je bezposrednio, bez `--device` ani edycji YAML.

```bash
python -m benchmark_runner run --config configs/experiments/GPU_AMD_IGPU/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_AMD_IGPU/image_classification_resnet50_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_AMD_IGPU/nlp_distilbert_imdb.yaml
```

**Przed pierwszym uruchomieniem przeczytaj [`../../../docs/setup_amd_igpu.md`](../../../docs/setup_amd_igpu.md)**
w calosci - w skrocie:

- Pomiar mocy dziala od razu, bez dodatkowej instalacji (zwykly sterownik jadra `amdgpu`).
- Compute (faktyczne odpalenie modelu na iGPU) wymaga **osobnego** venv z `torch+ROCm` -
  nie da sie tego pogodzic w jednym srodowisku z `torch+CUDA` (dyskretne GPU NVIDIA na tej
  samej maszynie, jesli jest, patrz `../GPU_NVIDIA/`). Czesto potrzebna jest tez zmienna
  `HSA_OVERRIDE_GFX_VERSION` - iGPU AMD w laptopach rzadko sa oficjalnie wspierane przez
  ROCm wprost.

**Status: faza `inference` dla `image_classification` ZWERYFIKOWANA** na maszynie
testowej "Ola" (Ryzen 7 5800HS, iGPU Radeon Vega, architektura `gfx90c`) - dziala z
`torch+rocm6.2` w osobnym venv i `HSA_OVERRIDE_GFX_VERSION=9.0.0` (na tym buildzie ROCm
rocBLAS nie ma natywnej biblioteki Tensile dla `gfx90c`, ale override na `gfx900` dziala
poprawnie). `nlp_distilbert_imdb.yaml` nie zostal jeszcze przetestowany osobno - powinien
dzialac tym samym mechanizmem, ale nie zakladaj tego bez wlasnej weryfikacji.

**Faza `train` jest wylaczona we wszystkich configach w tym folderze - celowo**, dopoki
ktos jej nie zweryfikuje na prawdziwym sprzecie (dluzsza, bardziej wymagajaca sciezka niz
inference, wiekszy koszt nieudanej probki calej serii). Jesli zweryfikujesz `train` na
swoim sprzecie, dopisz `"train"` do `phases` w odpowiednim pliku i zaktualizuj ten
README oraz `docs/setup_amd_igpu.md`.

**Brak `llm_inference_quantized.yaml` w tym folderze — celowo**, z tego samego powodu co
w `../GPU_AMD/README.md` i `../GPU_NVIDIA/README.md`: backend ONNX Runtime dla
`llm_inference` dziala obecnie tylko na CPU. Uzyj `../CPU/llm_inference_quantized.yaml`
na tej samej maszynie.

**Brak `int8` w `precisions` — z tego samego powodu**: dynamiczna kwantyzacja PyTorch
dziala tylko na CPU. Uzyj odpowiedniego configu z `../CPU/`.

**`measurement_scope` jest zawsze `whole_system`** dla tego device_type, niezaleznie od
tego, ktore zrodlo mocy faktycznie zostalo uzyte (`amdgpu_igpu_native` czy fallback
`amd_igpu_whole_system`) - odczyt `amdgpu_igpu_native` to `PPT` (Package Power Tracking),
pobor mocy calego SoC (CPU+iGPU razem), nie samego bloku graficznego. Nie porownuj tych
wynikow bezposrednio z `device_only` (CPU/CUDA/ROCm dyskretne w tym projekcie) bez tego
zastrzezenia - patrz [`../../../docs/measurement_methodology.md`](../../../docs/measurement_methodology.md).

Weryfikacja przed pierwszym uruchomieniem:

```bash
python scripts/check_device_availability.py
```

Powinno wypisac `amdgpu_igpu_native: DOSTEPNE` (pomiar mocy, dziala zawsze) i - jesli masz
aktywny venv z `torch+ROCm` - `amd_igpu: <nazwa karty> (backend=pytorch, ...)` w sekcji
"Wykryte urzadzenia obliczeniowe" (compute).
