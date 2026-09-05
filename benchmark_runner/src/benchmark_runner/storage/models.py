"""Dataclasses odzwierciedlajace tabele SQL (db/migrations/0001_init_schema.sql)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class DeviceRecord:
    machine_name: str
    device_type: str
    device_label: str
    vendor: str | None = None
    cpu_model: str | None = None
    gpu_model: str | None = None
    os_name: str | None = None
    os_version: str | None = None
    driver_version: str | None = None
    total_ram_gb: float | None = None
    notes: str | None = None
    # 'discrete' | 'integrated' | None (cpu/npu, lub GPU o nierozpoznanej etykiecie) - patrz
    # devices/gpu_classification.py. Heurystyka po nazwie urzadzenia, nie API sprzetowe.
    gpu_category: str | None = None


@dataclass
class RunRecord:
    device_id: str
    task: str
    model_name: str
    phase: str
    precision: str
    config_hash: str
    config_snapshot: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    duration_s: float
    dataset: str | None = None
    batch_size: int | None = None
    num_epochs: int | None = None
    num_iterations: int | None = None
    measurement_scope: str = "device_only"
    warmup_done: bool = True
    status: str = "completed"
    error_message: str | None = None
    throughput_samples_per_s: float | None = None
    avg_temperature_c: float | None = None
    # Grupuja powtorzenia (repetitions, patrz config/schema.py) tej samej kombinacji
    # parametrow - run_stats_summary agreguje po repetition_group_id.
    repetition_group_id: str | None = None
    repetition_index: int | None = None


@dataclass
class PowerSampleRecord:
    run_id: str
    timestamp: datetime
    source: str
    watts: float | None
    temperature_c: float | None = None


@dataclass
class RunSummaryRecord:
    run_id: str
    energy_joules_software: float | None = None
    energy_joules_smart_plug: float | None = None
    power_discrepancy_pct: float | None = None
    energy_per_sample_joules: float | None = None
    energy_per_epoch_joules: float | None = None
    avg_power_watts: float | None = None
    peak_power_watts: float | None = None
    flops_per_sample: float | None = None
    flops_per_watt: float | None = None
    samples_processed: int | None = None
    # Liczba probek mocy uzytych do policzenia avg_power_watts/peak_power_watts - wskaznik
    # wiarygodnosci tych metryk (mniej probek = mniej wiarygodna srednia), zwlaszcza przy
    # bardzo krotkich przebiegach - patrz metrics/aggregation.py::RunSummaryMetrics.
    power_samples_count: int | None = None
    # 'success' | 'partial' | 'failed' | None (None gdy precision != 'int8') - patrz
    # utils/quantization.py. 'partial' = np. tylko warstwy Linear skwantyzowane, Conv2d
    # bez wsparcia dynamicznej kwantyzacji w PyTorch pozostaly fp32.
    quantization_status: str | None = None
    # Dokladnosc klasyfikacji na zbiorze testowym (image_classification/nlp_sentiment) -
    # z TaskStepResult.accuracy, patrz core/base_task.py. None dla zadan bez metryki
    # dokladnosci (llm_inference) lub gdy faza='train' bez pelnego przejscia po zbiorze
    # testowym. Uzywane m.in. do oceny, czy kwantyzacja INT8 obniza jakosc modelu
    # (patrz dashboard "Wplyw parametrow" i docs/measurement_methodology.md).
    accuracy: float | None = None


@dataclass
class RunStatsSummaryRecord:
    """Agregacja statystyczna (Poprawka 2) po wszystkich powtorzeniach jednej kombinacji
    parametrow, liczona w Pythonie po stronie runnera (orchestrator._finalize_repetition_group)
    z zapisanych juz run_summary.duration_s/energy_joules_software - nie widok SQL."""

    repetition_group_id: str
    task: str
    model_name: str
    device_id: str
    precision: str
    phase: str
    n_repetitions: int
    batch_size: int | None = None
    mean_duration_s: float | None = None
    std_duration_s: float | None = None
    mean_energy_joules: float | None = None
    std_energy_joules: float | None = None
    coefficient_of_variation_energy: float | None = None
