# Configi dla PC z GPU NVIDIA

Wszystkie configi w tym folderze maja `devices: [cuda]` ustawione na sztywno -
uruchamiaj je bezposrednio, bez `--device` ani edycji YAML.

```bash
python -m benchmark_runner run --config configs/experiments/GPU_NVIDIA/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_NVIDIA/image_classification_resnet50_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_NVIDIA/nlp_distilbert_imdb.yaml
```

**Brak `llm_inference_quantized.yaml` w tym folderze — celowo.** Backend ONNX Runtime
uzywany przez zadanie `llm_inference` dziala obecnie tylko na CPU execution provider
(patrz `tasks/llm_inference/backends/onnxruntime_backend.py`) - config z `device: cuda`
dla tego zadania bylby mylacy, bo i tak liczylby na CPU, a etykietowalby wynik w bazie
jako "cuda". Uzyj `../CPU/llm_inference_quantized.yaml` na tej samej maszynie.

**Brak `int8` w `precisions` obu configow `image_classification_*` i
`nlp_distilbert_imdb.yaml` w tym folderze — z tego samego powodu.** Dynamiczna
kwantyzacja PyTorch (`utils/quantization.py`) dziala tylko na CPU - uzyj
odpowiedniego configu z `../CPU/` zeby zmierzyc `int8`.

Wymagania instalacyjne: patrz [`../../../docs/setup_windows_gpu.md`](../../../docs/setup_windows_gpu.md).
