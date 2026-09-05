# Configi dla maszyny z Intel GPU (zintegrowane Iris Xe/Arc lub dyskretne Arc)

Config w tym folderze ma `devices: [intel_gpu_openvino]` ustawione na sztywno - uruchamiaj
bezposrednio, bez `--device` ani edycji YAML:

```bash
python -m benchmark_runner run --config configs/experiments/GPU_INTEL/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_INTEL/image_classification_resnet50_cifar10.yaml
```

**Realnie dziala tu tylko `image_classification`.** `nlp_distilbert_imdb.yaml` jest w
folderze, ale jako config przygotowany na przyszlosc — sam OpenVINO obsluguje modele NLP
na Intel GPU, natomiast `NlpSentimentTask` nie ma jeszcze wpietej sciezki OpenVINO/ONNX
(uzywa wylacznie backendu `pytorch`, patrz naglowek
`src/benchmark_runner/tasks/nlp_sentiment/task.py`). Do czasu domkniecia tej sciezki
runner pominie/przerwie przebieg dla `nlp_sentiment` na `intel_gpu_openvino`. `llm_inference`
z tego samego powodu nie jest tu dostepne — uzyj `../CPU/`.

```bash
# BLOKOWANE do czasu wpiecia sciezki OpenVINO w NlpSentimentTask:
# python -m benchmark_runner run --config configs/experiments/GPU_INTEL/nlp_distilbert_imdb.yaml
```

**Kategoria dyskretne/zintegrowane jest wykrywana automatycznie** przy starcie (patrz
`devices/intel_gpu.py`) przez natywna wlasciwosc OpenVINO `DEVICE_TYPE` — nie musisz nic
konfigurowac w YAML. Wynik trafia do kolumny `devices.gpu_category` w Supabase i jest
widoczny w dashboardzie ("GPU dyskretne vs zintegrowane").

Pamietaj: ta sciezka wspiera tylko inferencje (nie trening), a wiarygodny pomiar mocy
programowej (nie watomierza) wymaga pracy na baterii — dokladnie jak dla NPU, patrz
[`../../../docs/setup_npu_intel.md`](../../../docs/setup_npu_intel.md) i
[`../../../docs/measurement_methodology.md`](../../../docs/measurement_methodology.md).

Weryfikacja przed pierwszym uruchomieniem:

```bash
python scripts/check_device_availability.py
```

Powinno wypisac `intel_gpu_openvino: DOSTEPNE`. Jesli nie — sprawdz, czy `openvino` widzi
GPU (`python -c "from openvino import Core; print(Core().available_devices)"` powinno
zawierac `'GPU'`) i czy sterownik graficzny Intela jest aktualny.
