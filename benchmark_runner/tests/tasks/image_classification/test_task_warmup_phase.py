"""Testy warmup() na sciezce pytorch (Blad 1 - warmup() wywolywal _train_step()
bezwarunkowo, niezaleznie od run_spec.phase). _train_step() robi model.train() +
backward/optimizer.step() - na phase='inference' z batch_size=1 to wywalalo sie na
BatchNorm ('Expected more than 1 value per channel when training'), mimo ze wlasciwa
inferencja (eval(), running_mean/var) dziala z batch_size=1 bez problemu. Blad byl wiec
calkowicie niezwiazany z rzeczywista inferencja - to warmup psul phase='inference'."""

from __future__ import annotations

from benchmark_runner.config.schema import ExperimentConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.tasks.image_classification import onnx_export
from benchmark_runner.tasks.image_classification.onnx_export import save_reference_checkpoint
from benchmark_runner.tasks.image_classification.models import build_model
from benchmark_runner.tasks.image_classification.task import ImageClassificationTask
from benchmark_runner.utils.system_info import SystemInfo


def _make_run_context(*, phase: str, batch_size: int, model: str = "mobilenet_v3") -> RunContext:
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=[model],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[batch_size],
        phases=[phase],
    )
    run_spec = experiment.expand_matrix()[0]
    device = DeviceProfile(device_type="cpu", label="Test CPU", vendor=None, torch_device_str="cpu", backend="pytorch")
    system_info = SystemInfo(machine_name="test-machine", os_name="Linux", os_version="1", cpu_model="Test CPU", total_ram_gb=16.0)
    return RunContext(run_spec=run_spec, device=device, system_info=system_info)


def test_warmup_and_inference_with_batch_size_1_does_not_crash_on_batchnorm(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    save_reference_checkpoint(build_model("mobilenet_v3"), "mobilenet_v3")

    run_context = _make_run_context(phase="inference", batch_size=1)
    task = ImageClassificationTask(run_context)
    task.prepare()

    task.warmup(3)  # przed poprawka: RuntimeError z BatchNorm ("Expected more than 1 value...")
    result = task.run_inference_pass()

    assert result.samples_processed > 0
    task.teardown()


def test_warmup_on_inference_phase_leaves_model_in_eval_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_export, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    save_reference_checkpoint(build_model("mobilenet_v3"), "mobilenet_v3")

    run_context = _make_run_context(phase="inference", batch_size=8)
    task = ImageClassificationTask(run_context)
    task.prepare()

    task.warmup(2)

    assert task._model.training is False, "warmup na phase='inference' musi zostawic model w eval(), nie train()"


def test_warmup_on_train_phase_still_trains(monkeypatch):
    # Regresja odwrotna - upewnia sie ze naprawa Bledu 1 NIE zepsula normalnego treningu.
    run_context = _make_run_context(phase="train", batch_size=8)
    task = ImageClassificationTask(run_context)
    task.prepare()

    initial_weight = task._model.classifier[-1].weight.detach().clone()
    task.warmup(2)

    assert task._model.training is True, "warmup na phase='train' nadal musi zostawic model w train()"
    assert not (task._model.classifier[-1].weight.detach() == initial_weight).all(), (
        "warmup na phase='train' musi realnie zaktualizowac wagi (backward + optimizer.step())"
    )
