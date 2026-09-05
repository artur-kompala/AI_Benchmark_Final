# Configi dla PC z GPU AMD

Wszystkie configi w tym folderze maja `devices: [rocm]` ustawione na sztywno oraz
`power_meter.gpu_sources: [rocm_smi]` - uruchamiaj je bezposrednio, bez `--device`
ani edycji YAML.

```bash
python -m benchmark_runner run --config configs/experiments/GPU_AMD/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_AMD/image_classification_resnet50_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/GPU_AMD/nlp_distilbert_imdb.yaml
```

**Uwaga**: oficjalne wsparcie `torch+ROCm` na Windows jest ograniczone - moze wymagac
WSL2. Zweryfikuj to jako pierwszy krok na swojej maszynie, patrz
[`../../../docs/setup_windows_gpu.md`](../../../docs/setup_windows_gpu.md).

**Brak `llm_inference_quantized.yaml` w tym folderze — celowo**, z tego samego powodu
co w `../GPU_NVIDIA/README.md`: backend ONNX Runtime dla `llm_inference` dziala
obecnie tylko na CPU. Uzyj `../CPU/llm_inference_quantized.yaml` na tej samej maszynie.

**Brak `int8` w `precisions` configow `image_classification_*`/`nlp_distilbert_imdb.yaml`
w tym folderze — z tego samego powodu**: dynamiczna kwantyzacja PyTorch dziala tylko
na CPU. Uzyj odpowiedniego configu z `../CPU/`.
