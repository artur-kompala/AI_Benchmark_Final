"""Zadanie klasyfikacji obrazow na CIFAR-10.

Dwie sciezki wykonania, wybierane wg run_context.device.backend:
- 'pytorch' (CPU/CUDA/ROCm): trening natywnie w PyTorch od losowej inicjalizacji (mierzymy
  dynamike/koszt treningu). Inferencja NATOMIAST wczytuje ten sam checkpoint referencyjny co
  sciezka onnx nizej (load_reference_checkpoint) - bez tego accuracy z inferencji na CPU/CUDA/
  ROCm bylaby rownie bez sensu jak byla wczesniej na NPU/Intel GPU (patrz onnx_export.py).
- 'openvino' (NPU lub Intel GPU - patrz devices/npu_openvino.py, devices/intel_gpu.py) /
  'onnxruntime' (NPU przez DirectML): TYLKO inferencja - model wczytuje wytrenowany
  checkpoint referencyjny (load_reference_checkpoint) i jest eksportowany do ONNX,
  uruchamiany przez OpenVINO (device="NPU" albo "GPU", patrz _OPENVINO_DEVICE_STRINGS
  nizej) lub ONNX Runtime z DirectML. Zadne z tych urzadzen nie wspiera treningu w tym
  ekosystemie, wiec faza 'train' konczy sie czytelnym bledem zamiast proby wykonania.

Accuracy referencyjna (obie sciezki, faza inference): liczona RAZ na PELNYM dostepnym
zbiorze testowym (bez cyklicznego zawijania, niezaleznie od batch_size/num_iterations
przebiegu wydajnosciowego) i zbuforowana w pamieci per (model, precision, device) na czas
zycia procesu - patrz _REFERENCE_ACCURACY_CACHE, _get_reference_accuracy() nizej. Wczesniej
accuracy bylo liczone na tej samej, zmiennej probce co pomiar wydajnosci (sterowanej przez
batch_size x num_iterations + cycling dataloadera), przez co roslo monotonicznie wraz z
samples_processed zamiast byc stabilna wlasciwoscia wag.
"""

from __future__ import annotations

import copy
import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from benchmark_runner.core.base_task import BaseTask, TaskStepResult
from benchmark_runner.core.registry import register_task
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.metrics.flops import estimate_flops
from benchmark_runner.tasks.image_classification.data import get_cifar10_dataloaders
from benchmark_runner.tasks.image_classification.models import build_model
from benchmark_runner.tasks.image_classification.onnx_export import export_to_onnx, load_reference_checkpoint
from benchmark_runner.utils.cycling_dataloader import cycling_batches
from benchmark_runner.utils.quantization import quantize_dynamic_int8

logger = logging.getLogger(__name__)

CIFAR10_INPUT_SHAPE = (3, 32, 32)

# int4 wymaga kwantyzacji NNCF (OpenVINO) - nie zaimplementowanej jeszcze, stad brak tej
# wartosci na obu sciezkach. int8 na sciezce pytorch = dynamiczna kwantyzacja (torch.
# quantization.quantize_dynamic, patrz utils/quantization.py i _prepare_pytorch ponizej),
# TYLKO na CPU (kernele FBGEMM/QNNPACK sa CPU-only) i TYLKO inferencja (orchestrator.
# _filter_unsupported_combos pomija int8+train przed wykonaniem, wiec self._backend=='pytorch'
# + precision=='int8' oznacza tu zawsze faze inference) - na sciezce onnx (NPU) int8 nadal
# spada do fp32 z ostrzezeniem (NNCF planowane w kolejnej iteracji).
_PYTORCH_SUPPORTED_PRECISIONS = {"fp32", "fp16", "int8"}
_ONNX_SUPPORTED_PRECISIONS = {"fp32", "fp16"}

_ONNX_BACKENDS = {"openvino", "onnxruntime"}

# device_type -> device string OpenVINO (Core().compile_model(device_name=...)) - oba
# urzadzenia uzywaja backend='openvino' (patrz devices/npu_openvino.py, devices/intel_gpu.py),
# wiec trzeba rozroznic po faktycznie wykrytym device_type, nie zakladac zawsze "NPU".
_OPENVINO_DEVICE_STRINGS = {"npu_openvino": "NPU", "intel_gpu_openvino": "GPU"}

# Cache accuracy referencyjnej per (model, precision, device_type) na czas zycia procesu.
# Modul-scoped (NIE atrybut instancji) bo orchestrator tworzy nowa instancje Task dla
# KAZDEGO przebiegu (patrz core/orchestrator.py::_run_single) - gdyby cache zyl na self,
# znikalby po kazdym przebiegu i liczylibysmy accuracy ponownie dla kazdego batch_size/
# repetition tej samej kombinacji, mimo ze accuracy jest wlasciwoscia WAG, nie przebiegu.
_REFERENCE_ACCURACY_CACHE: dict[tuple[str, str, str], float] = {}

# Batch_size uzywany WYLACZNIE do liczenia accuracy referencyjnej - celowo niezalezny od
# run_spec.batch_size (ktory steruje pomiarem wydajnosci/mocy, nie accuracy).
_REFERENCE_ACCURACY_BATCH_SIZE = 64


@register_task("image_classification")
class ImageClassificationTask(BaseTask):
    def __init__(self, run_context: RunContext) -> None:
        super().__init__(run_context)
        self._backend = run_context.device.backend

        # stan sciezki pytorch (CPU/CUDA/ROCm)
        self._device: torch.device | None = None
        self._use_amp = False
        self._model: nn.Module | None = None
        self._optimizer: optim.Optimizer | None = None
        self._scaler: GradScaler | None = None
        self._criterion = nn.CrossEntropyLoss()
        self._train_loader = None

        # stan sciezki onnx (openvino/onnxruntime, NPU)
        self._inference_backend = None
        self._onnx_path: Path | None = None

        # wspolne (uzywane przez obie sciezki dla inferencji)
        self._test_loader = None

        # Accuracy referencyjna (faza inference, obie sciezki) - ustawiana w prepare(),
        # patrz _get_reference_accuracy()/_REFERENCE_ACCURACY_CACHE wyzej. None dla fazy
        # 'train' (tam accuracy liczone jest na biezaco per epoka, patrz run_train_epoch).
        self._reference_accuracy: float | None = None

        # ustawiane tylko gdy precision=='int8' i backend=='pytorch' (patrz _prepare_pytorch) -
        # dolaczane do TaskStepResult.extra, orchestrator zapisuje w run_summary.quantization_status.
        self._quantization_status: str | None = None

        # FLOPS dla jednego przejscia forward (batch=1), liczone raz w prepare() - patrz
        # estimate_flops_per_sample() nizej i metrics/flops.py.
        self._flops_per_sample: float | None = None

        if self._backend == "pytorch":
            self._device = torch.device(run_context.device.torch_device_str or "cpu")
            self._use_amp = run_context.run_spec.precision == "fp16" and self._device.type == "cuda"
            self._scaler = GradScaler(device=self._device.type, enabled=self._use_amp)

    def prepare(self) -> None:
        run_spec = self.run_context.run_spec

        if self._backend == "pytorch":
            self._prepare_pytorch(run_spec)
        elif self._backend in _ONNX_BACKENDS:
            if run_spec.precision not in _ONNX_SUPPORTED_PRECISIONS:
                logger.warning(
                    "Precyzja '%s' nie jest jeszcze wspierana na sciezce OpenVINO/ONNX Runtime "
                    "(wymaga kwantyzacji NNCF, planowane w kolejnej iteracji) - uruchamiam mimo to z fp32.",
                    run_spec.precision,
                )
            self._prepare_onnx(run_spec)
        else:
            raise NotImplementedError(f"Backend '{self._backend}' nie jest wspierany przez ImageClassificationTask")

    def _prepare_pytorch(self, run_spec) -> None:
        # Osobna, jednorazowa instancja CPU/fp32 WYLACZNIE do oszacowania FLOPS - niezalezna
        # od faktycznego urzadzenia/precyzji przebiegu (FLOPS to statyczna wlasciwosc
        # architektury; liczenie z modelu juz skwantyzowanego dawaloby zanizony wynik, bo
        # thop/fvcore nie rozpoznaja niestandardowych typow warstw DynamicQuantizedLinear).
        self._flops_per_sample = estimate_flops(build_model(run_spec.model), input_shape=CIFAR10_INPUT_SHAPE)

        model = build_model(run_spec.model)
        if run_spec.phase == "inference":
            # Tak jak sciezka OpenVINO/ONNX Runtime (_prepare_onnx nizej) - inferencja ma
            # mierzyc wydajnosc/moc WYTRENOWANEGO modelu, nie swiezo zainicjalizowanego.
            # Faza 'train' celowo zaczyna od losowych wag (mierzymy sama dynamike/koszt
            # treningu, nie koncowa dokladnosc). Rzuca ReferenceCheckpointNotFoundError z
            # czytelna instrukcja, jesli checkpoint nie istnieje - celowo nie lapiemy tego
            # wyjatku tutaj (patrz onnx_export.py).
            load_reference_checkpoint(model, run_spec.model, dataset=run_spec.dataset or "cifar10")
        model = model.to(self._device)

        if run_spec.precision == "int8":
            if self._device.type != "cpu":
                # Obronny fallback - configi z int8 sa celowo tylko w CPU/ (dynamiczna
                # kwantyzacja PyTorch dziala jedynie na CPU), ale --device moze to nadpisac.
                logger.warning(
                    "Precyzja int8 wymaga backendu CPU (dynamiczna kwantyzacja PyTorch nie dziala na %s) - "
                    "uruchamiam mimo to z fp32",
                    self._device.type,
                )
                self._quantization_status = "failed"
            else:
                model.eval()
                model, self._quantization_status = quantize_dynamic_int8(model)

        self._model = model
        self._train_loader, self._test_loader = get_cifar10_dataloaders(batch_size=run_spec.batch_size)
        self._optimizer = optim.Adam(self._model.parameters(), lr=run_spec.train.learning_rate)

        if run_spec.phase == "inference":
            self._reference_accuracy = self._get_reference_accuracy(self._model, run_spec)

    def _prepare_onnx(self, run_spec) -> None:
        # Ta sciezka (OpenVINO/ONNX Runtime na NPU/Intel GPU) nigdy sama nie trenuje -
        # MUSI wczytac wagi wytrenowane raz gdzie indziej (CPU/GPU_AMD/GPU_NVIDIA), inaczej
        # accuracy mierzone tutaj jest bez sensu (poziom przypadku). Rzuca
        # ReferenceCheckpointNotFoundError z czytelna instrukcja, jesli checkpoint nie
        # istnieje - celowo NIE lapiemy tego wyjatku tutaj (patrz onnx_export.py).
        pytorch_model = build_model(run_spec.model)
        load_reference_checkpoint(pytorch_model, run_spec.model, dataset=run_spec.dataset or "cifar10")
        pytorch_model.eval()
        self._flops_per_sample = estimate_flops(pytorch_model, input_shape=CIFAR10_INPUT_SHAPE)
        self._reference_accuracy = self._get_reference_accuracy(pytorch_model, run_spec)

        onnx_path = Path(tempfile.gettempdir()) / f"benchmark_runner_{run_spec.model}_{self.run_context.run_id}.onnx"
        export_to_onnx(pytorch_model, batch_size=run_spec.batch_size, input_shape=CIFAR10_INPUT_SHAPE, output_path=onnx_path)
        self._onnx_path = onnx_path

        # drop_last_test=True: self._test_loader tutaj karmi wylacznie sciezke OpenVINO/ONNX
        # Runtime (_run_inference_onnx/_warmup_onnx nizej), ktora dziala na modelu o
        # statycznym ksztalcie wejscia - kazdy batch MUSI miec dokladnie batch_size probek
        # (patrz cycling_batches w utils/cycling_dataloader.py).
        _, self._test_loader = get_cifar10_dataloaders(batch_size=run_spec.batch_size, drop_last_test=True)

        if self._backend == "openvino":
            from benchmark_runner.inference_backends.openvino_backend import OpenVinoInferenceBackend

            # Device string OpenVINO ('NPU' albo 'GPU') zalezny od faktycznie wykrytego
            # urzadzenia, NIE zawsze 'NPU' - backend='openvino' obsluguje teraz zarowno
            # npu_openvino jak i intel_gpu_openvino (patrz devices/intel_gpu.py).
            ov_device = _OPENVINO_DEVICE_STRINGS.get(self.run_context.device.device_type, "NPU")
            self._inference_backend = OpenVinoInferenceBackend(onnx_path, device=ov_device)
        else:  # onnxruntime
            from benchmark_runner.inference_backends.onnxruntime_backend import OnnxRuntimeInferenceBackend

            self._inference_backend = OnnxRuntimeInferenceBackend(onnx_path, provider="DmlExecutionProvider")

        self._inference_backend.compile()

    def _get_reference_accuracy(self, model: nn.Module, run_spec) -> float:
        """Accuracy referencyjna dla (model, precision, device) - patrz modul docstring i
        _REFERENCE_ACCURACY_CACHE. Liczona RAZ per proces, zbuforowana dla wszystkich
        batch_size/powtorzen tej samej kombinacji."""
        device_type = self.run_context.device.device_type
        key = (run_spec.model, run_spec.precision, device_type)

        cached = _REFERENCE_ACCURACY_CACHE.get(key)
        if cached is not None:
            logger.info(
                "Accuracy referencyjna dla model=%s precision=%s device=%s juz policzona w "
                "tym procesie (%.4f) - uzywam z cache zamiast liczyc ponownie",
                *key,
                cached,
            )
            return cached

        accuracy = self._compute_reference_accuracy(model)
        _REFERENCE_ACCURACY_CACHE[key] = accuracy
        logger.info(
            "Policzono accuracy referencyjna dla model=%s precision=%s device=%s: %.4f "
            "(pelny test set, batch_size=%d, bez cyklicznego zawijania) - bedzie uzyta dla "
            "wszystkich batch_size/powtorzen tej kombinacji zamiast liczenia od nowa",
            *key,
            accuracy,
            _REFERENCE_ACCURACY_BATCH_SIZE,
        )
        return accuracy

    def _compute_reference_accuracy(self, model: nn.Module) -> float:
        """Jednorazowy, pelny przebieg po CALYM dostepnym zbiorze testowym (bez cyklicznego
        zawijania i niezaleznie od batch_size/num_iterations przebiegu wydajnosciowego) -
        dziala na NIEZALEZNEJ kopii modelu (deepcopy) na CPU, zeby nie ruszac device/trybu
        (train/eval) obiektu uzywanego przez wlasciwy pomiar wydajnosci (self._model moze
        byc na CUDA/ROCm albo skwantyzowany int8 - przenoszenie/mutacja tego obiektu
        zepsulaby nastepujacy po tym pomiar)."""
        reference_model = copy.deepcopy(model).to("cpu").eval()
        _, reference_loader = get_cifar10_dataloaders(batch_size=_REFERENCE_ACCURACY_BATCH_SIZE, drop_last_test=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in reference_loader:
                outputs = reference_model(inputs)
                correct += int((outputs.argmax(dim=1) == targets).sum().item())
                total += inputs.size(0)
        return correct / total if total else 0.0

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, int]:
        assert self._model is not None and self._optimizer is not None
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        self._optimizer.zero_grad()
        with autocast(device_type=self._device.type, enabled=self._use_amp):
            outputs = self._model(inputs)
            loss = self._criterion(outputs, targets)

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        correct = int((outputs.argmax(dim=1) == targets).sum().item())
        return loss.item(), correct

    def warmup(self, n_iters: int) -> None:
        if self._backend == "pytorch":
            if self.run_context.run_spec.phase == "train":
                self._warmup_pytorch_train(n_iters)
            else:
                self._warmup_pytorch_inference(n_iters)
        else:
            self._warmup_onnx(n_iters)

    def _warmup_pytorch_train(self, n_iters: int) -> None:
        assert self._model is not None and self._train_loader is not None
        self._model.train()
        data_iter = iter(self._train_loader)
        for _ in range(n_iters):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self._train_loader)
                inputs, targets = next(data_iter)
            self._train_step(inputs, targets)

    def _warmup_pytorch_inference(self, n_iters: int) -> None:
        # Blad: warmup wywolywal bezwarunkowo _train_step() (model.train() + backward)
        # nawet na phase='inference' - z batch_size=1 wywalalo sie to na BatchNorm
        # ("Expected more than 1 value per channel when training"), mimo ze wlasciwa
        # inferencja (eval(), statystyki running_mean/var) dziala z batch_size=1 bez
        # problemu. Warmup musi robic DOKLADNIE to samo co pomiar wlasciwy - forward-only,
        # bez backward/optimizer.step() - zeby i nie psuc phase='inference', i realnie
        # rozgrzewac ta sama sciezke wykonania co potem mierzy run_inference_pass().
        assert self._model is not None and self._test_loader is not None
        self._model.eval()
        data_iter = iter(self._test_loader)
        with torch.no_grad(), autocast(device_type=self._device.type, enabled=self._use_amp):
            for _ in range(n_iters):
                try:
                    inputs, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._test_loader)
                    inputs, _ = next(data_iter)
                inputs = inputs.to(self._device)
                self._model(inputs)

    def _warmup_onnx(self, n_iters: int) -> None:
        assert self._inference_backend is not None and self._test_loader is not None
        for inputs, _ in cycling_batches(self._test_loader, n_iters):
            self._inference_backend.infer(inputs.numpy())

    def run_train_epoch(self) -> TaskStepResult:
        if self._backend != "pytorch":
            raise NotImplementedError(
                "Faza 'train' nie jest wspierana na urzadzeniach NPU (OpenVINO/ONNX Runtime "
                "obsluguja tylko inferencje) - usun 'train' z listy faz dla urzadzen NPU w configu"
            )

        assert self._model is not None and self._train_loader is not None
        self._model.train()

        total_samples = 0
        total_loss = 0.0
        correct = 0

        for inputs, targets in self._train_loader:
            batch_size = inputs.size(0)
            loss_value, batch_correct = self._train_step(inputs, targets)
            total_samples += batch_size
            total_loss += loss_value * batch_size
            correct += batch_correct

        return TaskStepResult(
            samples_processed=total_samples,
            loss=total_loss / total_samples if total_samples else None,
            accuracy=correct / total_samples if total_samples else None,
        )

    def run_inference_pass(self) -> TaskStepResult:
        if self._backend == "pytorch":
            return self._run_inference_pytorch()
        return self._run_inference_onnx()

    def _run_inference_pytorch(self) -> TaskStepResult:
        # accuracy NIE jest liczone tutaj z tej petli - patrz self._reference_accuracy
        # (_get_reference_accuracy w prepare()). Ta petla mierzy WYLACZNIE
        # wydajnosc/moc na num_iterations batchy; wczesniej liczyla accuracy na tej samej
        # zmiennej probce, przez co accuracy roslo wraz z samples_processed (Blad).
        assert self._model is not None and self._test_loader is not None
        self._model.eval()

        total_samples = 0
        num_iterations = self.run_context.run_spec.inference.num_iterations

        with torch.no_grad(), autocast(device_type=self._device.type, enabled=self._use_amp):
            for i, (inputs, _) in enumerate(self._test_loader):
                if i >= num_iterations:
                    break
                inputs = inputs.to(self._device)
                self._model(inputs)
                total_samples += inputs.size(0)

        extra = {"quantization_status": self._quantization_status} if self._quantization_status else {}
        return TaskStepResult(
            samples_processed=total_samples,
            accuracy=self._reference_accuracy,
            extra=extra,
        )

    def _run_inference_onnx(self) -> TaskStepResult:
        # accuracy NIE jest liczone tutaj z tej petli - patrz komentarz w
        # _run_inference_pytorch wyzej, dotyczy identycznie obu sciezek.
        assert self._inference_backend is not None and self._test_loader is not None

        total_samples = 0
        num_iterations = self.run_context.run_spec.inference.num_iterations

        for inputs, _ in cycling_batches(self._test_loader, num_iterations):
            self._inference_backend.infer(inputs.numpy())
            total_samples += inputs.shape[0]

        return TaskStepResult(
            samples_processed=total_samples,
            accuracy=self._reference_accuracy,
        )

    def estimate_flops_per_sample(self) -> float | None:
        return self._flops_per_sample

    def teardown(self) -> None:
        self._model = None
        self._optimizer = None
        self._train_loader = None
        self._test_loader = None
        self._inference_backend = None
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
        if self._onnx_path is not None:
            try:
                self._onnx_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Nie udalo sie usunac tymczasowego pliku ONNX %s", self._onnx_path, exc_info=True)
            self._onnx_path = None
