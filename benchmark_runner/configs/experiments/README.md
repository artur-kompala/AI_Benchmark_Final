# Configi eksperymentow - zorganizowane per urzadzenie

Configi sa pogrupowane w foldery odpowiadajace typowi sprzetu, zeby uruchamianie na
roznych maszynach nie wymagalo edycji YAML ani pamietania flagi `--device` - wybierz
folder odpowiadajacy maszynie, na ktorej jestes, i odpal plik o tej samej nazwie co
na innych maszynach:

- **[`CPU/`](CPU/)** - dowolna maszyna, `devices: [cpu]`. Jedyny folder z pelnym
  kompletem trzech zadan (`image_classification`, `nlp_sentiment`, `llm_inference`).
- **[`GPU_NVIDIA/`](GPU_NVIDIA/)** - PC z GPU NVIDIA, `devices: [cuda]`.
- **[`GPU_AMD/`](GPU_AMD/)** - PC z GPU AMD, `devices: [rocm]`.
- **[`GPU_INTEL/`](GPU_INTEL/)** - maszyna z Intel GPU (zintegrowane Iris Xe/Arc lub
  dyskretne Arc), `devices: [intel_gpu_openvino]`. Kategoria dyskretne/zintegrowane
  wykrywana automatycznie (OpenVINO `DEVICE_TYPE`). Tylko inferencja, jak `NPU/`.
- **[`NPU/`](NPU/)** - laptop z Intel Core Ultra, `devices: [npu_openvino]`.

Pliki o tej samej nazwie w roznych folderach maja **celowo identyczna** macierz
`batch_sizes`/`precisions`/`phases` (gdzie sprzet na to pozwala) - to wazne dla
poprawnego porownania urzadzen w pracy: te same kombinacje parametrow, jedyna
roznica to `devices`.

`GPU_NVIDIA/`, `GPU_AMD/` i `NPU/` **nie zawieraja** `llm_inference_quantized.yaml` -
backend ONNX Runtime uzywany przez to zadanie dziala obecnie tylko na CPU execution
provider, wiec config z innym urzadzeniem bylby mylacy (etykietowalby wynik w bazie
pod GPU/NPU, mimo ze liczyłby na CPU). `NPU/` nie zawiera tez `nlp_sentiment` - ta
sciezka nie jest jeszcze wspierana na NPU. Kazdy folder ma wlasny `README.md` z
detalami.

## Modele klasyfikacji obrazow: MobileNetV3 i ResNet-50

Kazdy folder ma DWA osobne configi klasyfikacji obrazow -
`image_classification_mobilenet_cifar10.yaml` i
`image_classification_resnet50_cifar10.yaml` - celowo w osobnych plikach, nie
polaczone w jeden config z dwoma modelami, zeby macierz `batch_sizes`/`precisions`
kazdego z nich mozna bylo dostosowac niezaleznie (ResNet-50 ma mniejsze
`num_epochs`/`repetitions` niz MobileNetV3 - patrz komentarz w
`CPU/image_classification_resnet50_cifar10.yaml` - ResNet-50 jest znaczaco
kosztowniejszy obliczeniowo).

## Powtorzenia pomiarow (`repetitions`)

Kazdy config ma pole `repetitions` (patrz `docs/measurement_methodology.md`,
"Powtorzenia pomiarow i statystyka") - liczba niezaleznych powtorzen kazdej
unikalnej kombinacji parametrow, do policzenia odchylenia standardowego/wariancji.
Domyslnie 3, obnizone do 2 dla kosztownych kombinacji treningowych (ResNet-50).

## Kwantyzacja INT8 poza LLM

`int8` w `precisions` dla `image_classification` i `nlp_sentiment` (dynamiczna
kwantyzacja PyTorch - patrz `utils/quantization.py`) jest CELOWO obecne **tylko**
w configach z folderu `CPU/` - dynamiczna kwantyzacja PyTorch dziala tylko na CPU
(kernele FBGEMM/QNNPACK), wiec config z `device: cuda`/`rocm` i `int8` liczylby sie
i tak na CPU, mylnie etykietujac wynik w bazie (ten sam powod co brak
`llm_inference_quantized.yaml` w `GPU_*/`/`NPU/`, wyzej). Dotyczy tylko fazy
inference - kombinacje `int8`+`train` sa automatycznie pomijane przez runner, nie
trzeba ich usuwac recznie z macierzy configu.

## Watomierz fizyczny

Wszystkie configi maja domyslnie `power_meter.smart_plug.backend: "tuya"`. Jesli
dana maszyna nie ma skonfigurowanego watomierza w `.env` (`TUYA_DEVICE_ID`/
`TUYA_LOCAL_KEY`/`TUYA_IP_ADDRESS`), runner automatycznie i bezpiecznie wraca do
trybu manualnego - nie trzeba edytowac configu. Patrz
[`../../../docs/setup_smart_plug.md`](../../../docs/setup_smart_plug.md).

## Nadpisanie urzadzenia mimo wszystko

Jesli mimo wszystko chcesz wymusic inne urzadzenie niz to wpisane w configu (np. do
szybkiego testu), flaga CLI `--device` ma pierwszenstwo nad polem `devices` w YAML:

```bash
python -m benchmark_runner run --config configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml --device cuda
```

Patrz tez [`device_overrides/example_force_cpu.yaml`](device_overrides/example_force_cpu.yaml)
po przyklad tego mechanizmu.
