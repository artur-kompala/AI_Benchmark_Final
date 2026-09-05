from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_runner.config.loader import ConfigError, load_config
from benchmark_runner.config.schema import ExperimentConfig

_EXAMPLE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "CPU"
    / "image_classification_mobilenet_cifar10.yaml"
)


def test_load_valid_example_config():
    config = load_config(_EXAMPLE_CONFIG)
    assert config.task == "image_classification"
    assert config.models == ["mobilenet_v3"]
    assert config.precisions == ["fp32", "fp16", "int8"]

    assert config.repetitions == 3
    assert config.batch_sizes == [1, 4, 32, 64]

    run_specs = config.expand_matrix()
    # models(1) x devices(1) x precisions(3) x batch_sizes(4) x phases(2) = 24 kombinacje,
    # kazda powielona repetitions=3 razy pod rzad (patrz config/schema.py:expand_matrix) = 72.
    # (expand_matrix() sam NIE filtruje int8+train - to robi orchestrator.run_from_config,
    # patrz _filter_unsupported_combos w core/orchestrator.py)
    assert len(run_specs) == 72

    # powtorzenia tej samej kombinacji dziela repetition_group_id, ale maja rozne seedy/indeksy
    first_group = run_specs[:3]
    assert len({r.repetition_group_id for r in first_group}) == 1
    assert [r.repetition_index for r in first_group] == [1, 2, 3]
    assert [r.seed for r in first_group] == [42, 43, 44]


def test_load_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_invalid_yaml_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("task: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_file)


def test_load_missing_required_field_raises_config_error(tmp_path):
    bad_file = tmp_path / "missing_field.yaml"
    bad_file.write_text("experiment_name: test\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_file)


def test_empty_list_field_raises_config_error(tmp_path):
    bad_file = tmp_path / "empty_list.yaml"
    bad_file.write_text(
        """
experiment_name: test
task: image_classification
models: []
precisions: [fp32]
batch_sizes: [32]
phases: [train]
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(bad_file)


# ============================================================
# Izolacja RunConfig w expand_matrix() - Blad: przekazanie tego samego obiektu
# self.train/self.inference/self.warmup/self.power_meter (bez .model_copy) do wielu
# RunConfig sprawialo, ze pydantic uzywal TEJ SAMEJ instancji przez referencje we
# wszystkich - mutacja w jednym przebiegu (core/orchestrator.py::
# _maybe_scale_up_num_iterations podnoszace run_spec.inference.num_iterations dla
# KONKRETNEGO batch_size) natychmiast "przeciekala" do WSZYSTKICH innych RunConfig w tej
# samej macierzy, w tym o zupelnie innym batch_size. Potwierdzone empirycznie: identyczne
# przeskalowane num_iterations (1353) dla kazdego batch_size w configu, mimo drastycznie
# roznego czasu iteracji miedzy nimi.
# ============================================================


def _make_multi_batch_experiment(batch_sizes: list[int]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=batch_sizes,
        phases=["inference"],
        repetitions=1,
    )


def test_expand_matrix_gives_each_run_spec_independent_inference_config():
    run_specs = _make_multi_batch_experiment([1, 4, 32, 64]).expand_matrix()
    assert len(run_specs) == 4

    assert run_specs[0].inference is not run_specs[1].inference
    assert run_specs[0].warmup is not run_specs[1].warmup
    assert run_specs[0].power_meter is not run_specs[1].power_meter
    assert run_specs[0].train is not run_specs[1].train


def test_mutating_one_run_specs_num_iterations_does_not_leak_to_others():
    run_specs = _make_multi_batch_experiment([1, 4, 32, 64]).expand_matrix()
    original_values = [r.inference.num_iterations for r in run_specs]

    # Symuluje dokladnie to, co robi _maybe_scale_up_num_iterations/retry loop w
    # orchestrator.py dla JEDNEGO konkretnego przebiegu (batch_size=1).
    run_specs[0].inference.num_iterations = 1353

    assert run_specs[0].inference.num_iterations == 1353
    for r in run_specs[1:]:
        assert r.inference.num_iterations == original_values[1]  # niezmienione, wlasna kopia


def test_mutating_repetitions_of_same_combo_does_not_leak_across_repetitions():
    # Repetitions tej samej kombinacji (ten sam repetition_group_id) tez musza miec
    # niezalezne obiekty - skalowanie w powtorzeniu 1 nie powinno wplywac na powtorzenie 2.
    experiment = ExperimentConfig(
        experiment_name="test",
        task="image_classification",
        dataset="cifar10",
        models=["mobilenet_v3"],
        devices=["auto"],
        precisions=["fp32"],
        batch_sizes=[1],
        phases=["inference"],
        repetitions=3,
    )
    run_specs = experiment.expand_matrix()
    assert len(run_specs) == 3

    run_specs[0].inference.num_iterations = 500
    assert run_specs[1].inference.num_iterations != 500
    assert run_specs[2].inference.num_iterations != 500
