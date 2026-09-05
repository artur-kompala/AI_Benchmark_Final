# Configi dla laptopa z Intel Core Ultra (NPU)

Config w tym folderze ma `devices: [npu_openvino]` ustawione na sztywno - uruchamiaj
bezposrednio, bez `--device` ani edycji YAML:

```bash
python -m benchmark_runner run --config configs/experiments/NPU/image_classification_mobilenet_cifar10.yaml
python -m benchmark_runner run --config configs/experiments/NPU/image_classification_resnet50_cifar10.yaml
```

**Tylko `image_classification` jest tu dostepne — celowo.** `nlp_sentiment` i
`llm_inference` nie wspieraja jeszcze sciezki NPU (obie rzucaja czytelny
`NotImplementedError` przy proibie uruchomienia na `npu_openvino`/`npu_directml` -
patrz `tasks/nlp_sentiment/task.py` i `tasks/llm_inference/backends/onnxruntime_backend.py`).
Uzyj `../CPU/nlp_distilbert_imdb.yaml` i `../CPU/llm_inference_quantized.yaml` na tej
samej maszynie (laptop z NPU ma tez CPU).

Pamietaj: NPU wspiera tylko inferencje (nie trening), a wiarygodny pomiar mocy
programowej (nie watomierza) wymaga pracy na baterii - patrz
[`../../../docs/setup_npu_intel.md`](../../../docs/setup_npu_intel.md).
