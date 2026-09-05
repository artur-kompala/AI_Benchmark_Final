"""Testy accuracy referencyjnej (Problem 2 - accuracy nadal sprzezone z batch_size/liczba
przetworzonych probek mimo wczesniejszej poprawki checkpointu). Potwierdzone empirycznie:
accuracy roslo monotonicznie wraz z samples_processed w tej samej grupie modelu (mobilenet:
0.32 przy 50 probkach -> 0.357 przy 3200 probkach), bo bylo liczone na tej samej, zmiennej
probce co pomiar wydajnosci. Accuracy referencyjna ma byc liczona RAZ na (model, precision,
device), na PELNYM dostepnym zbiorze testowym, i zbuforowana w pamieci procesu."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.image_classification import task as task_module
from benchmark_runner.tasks.image_classification.task import ImageClassificationTask
from benchmark_runner.utils.system_info import SystemInfo


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _FakeInferenceBackend:
    def infer(self, input_array):
        return [[1.0] + [0.0] * 9] * input_array.shape[0]


class _FakeCyclingSource:
    """Loader-podobny obiekt zgodny z cycling_batches() (wymaga drop_last=True, __len__,
    __iter__ zwracajacy nieskonczona (dla testu: dlugie) sekwencje batchy o stalym batch_size)."""

    def __init__(self, batch_size: int, dataset_len: int = 1000) -> None:
        self.batch_size = batch_size
        self.drop_last = True
        self._n_batches = dataset_len // batch_size

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        for _ in range(self._n_batches):
            yield torch.zeros(self.batch_size, 3, 32, 32), torch.zeros(self.batch_size, dtype=torch.long)


class _FakePlainSource:
    """Loader-podobny obiekt dla sciezki pytorch (_run_inference_pytorch) - zwykle
    enumerate(loader), bez wymogu drop_last."""

    def __init__(self, batch_size: int, n_batches: int = 1000) -> None:
        self._batch_size = batch_size
        self._n_batches = n_batches

    def __iter__(self):
        for _ in range(self._n_batches):
            yield torch.zeros(self._batch_size, 3, 32, 32), torch.zeros(self._batch_size, dtype=torch.long)


@pytest.fixture(autouse=True)
def _clear_reference_accuracy_cache():
    task_module._REFERENCE_ACCURACY_CACHE.clear()
    yield
    task_module._REFERENCE_ACCURACY_CACHE.clear()


def _make_run_context(
    *, backend: str, device_type: str, model: str = "mobilenet_v3", batch_size: int = 8, precision: str = "fp32"
) -> RunContext:
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=[model],
        devices=["auto"],
        precisions=[precision],
        batch_sizes=[batch_size],
        phases=["inference"],
    )
    run_spec = experiment.expand_matrix()[0]
    device = DeviceProfile(device_type=device_type, label="Test Device", vendor=None, torch_device_str="cpu", backend=backend)
    system_info = SystemInfo(machine_name="test-machine", os_name="Linux", os_version="1", cpu_model="Test CPU", total_ram_gb=16.0)
    return RunContext(run_spec=run_spec, device=device, system_info=system_info)


# ============================================================
# Cache: RAZ na (model, precision, device), nie ponownie per batch_size/repetition.
# ============================================================


def test_get_reference_accuracy_computes_once_and_reuses_cache(monkeypatch):
    run_context = _make_run_context(backend="pytorch", device_type="cpu")
    task = ImageClassificationTask(run_context)

    call_count = 0

    def _fake_compute(self, model):
        nonlocal call_count
        call_count += 1
        return 0.42

    monkeypatch.setattr(ImageClassificationTask, "_compute_reference_accuracy", _fake_compute)

    model = _TinyModel()
    first = task._get_reference_accuracy(model, run_context.run_spec)
    second = task._get_reference_accuracy(model, run_context.run_spec)

    assert first == pytest.approx(0.42)
    assert second == pytest.approx(0.42)
    assert call_count == 1, "druga wywolanie z tym samym kluczem musi trafic do cache"


def test_cache_is_keyed_by_model_precision_device_not_shared_across_them(monkeypatch):
    calls = []

    def _fake_compute(self, model):
        calls.append(1)
        return 0.5 + 0.01 * len(calls)

    monkeypatch.setattr(ImageClassificationTask, "_compute_reference_accuracy", _fake_compute)

    ctx_mobilenet = _make_run_context(backend="pytorch", device_type="cpu", model="mobilenet_v3")
    ctx_resnet = _make_run_context(backend="pytorch", device_type="cpu", model="resnet50")
    ctx_cuda = _make_run_context(backend="pytorch", device_type="cuda", model="mobilenet_v3")

    task_mobilenet = ImageClassificationTask(ctx_mobilenet)
    task_resnet = ImageClassificationTask(ctx_resnet)
    task_cuda = ImageClassificationTask(ctx_cuda)

    acc_mobilenet = task_mobilenet._get_reference_accuracy(_TinyModel(), ctx_mobilenet.run_spec)
    acc_resnet = task_resnet._get_reference_accuracy(_TinyModel(), ctx_resnet.run_spec)
    acc_cuda = task_cuda._get_reference_accuracy(_TinyModel(), ctx_cuda.run_spec)

    assert len(calls) == 3, "trzy rozne klucze (model/device) -> trzy niezalezne obliczenia"
    assert len({acc_mobilenet, acc_resnet, acc_cuda}) == 3


# ============================================================
# Decoupling: accuracy zwracane z inferencji NIE zalezy od num_iterations/batch_size
# petli wydajnosciowej - jest zawsze self._reference_accuracy ustawione w prepare().
# ============================================================


def test_run_inference_onnx_accuracy_independent_of_num_iterations():
    run_context = _make_run_context(backend="openvino", device_type="npu_openvino", batch_size=4)
    task = ImageClassificationTask(run_context)
    task._inference_backend = _FakeInferenceBackend()
    task._test_loader = _FakeCyclingSource(batch_size=4, dataset_len=40)  # tylko 10 pelnych batchy
    task._reference_accuracy = 0.777

    run_context.run_spec.inference.num_iterations = 3
    result_few = task._run_inference_onnx()

    run_context.run_spec.inference.num_iterations = 30  # wiecej niz dostepnych batchy -> zawija
    result_many = task._run_inference_onnx()

    assert result_few.samples_processed != result_many.samples_processed
    assert result_few.accuracy == pytest.approx(0.777)
    assert result_many.accuracy == pytest.approx(0.777)
    assert result_few.accuracy == result_many.accuracy


def test_run_inference_pytorch_accuracy_independent_of_num_iterations():
    run_context = _make_run_context(backend="pytorch", device_type="cpu", batch_size=4)
    task = ImageClassificationTask(run_context)
    task._device = torch.device("cpu")
    task._use_amp = False
    task._model = _TinyModel()
    task._test_loader = _FakePlainSource(batch_size=4, n_batches=1000)
    task._reference_accuracy = 0.512

    run_context.run_spec.inference.num_iterations = 2
    result_few = task._run_inference_pytorch()

    run_context.run_spec.inference.num_iterations = 40
    result_many = task._run_inference_pytorch()

    assert result_few.samples_processed != result_many.samples_processed
    assert result_few.accuracy == pytest.approx(0.512)
    assert result_many.accuracy == pytest.approx(0.512)
