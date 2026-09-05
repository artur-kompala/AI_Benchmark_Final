"""Agregacja surowych probek mocy w metryki pochodne (energia, throughput, ...)."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_runner.power.base_power_meter import PowerSample


def integrate_energy_joules(samples: list[PowerSample]) -> float | None:
    """Calkowanie trapezoidalne mocy [W] po czasie [s] -> energia [J].
    Probki z watts=None sa pomijane. Zwraca None gdy mniej niz 2 uzyteczne probki."""
    usable = [s for s in samples if s.watts is not None]
    if len(usable) < 2:
        return None
    usable = sorted(usable, key=lambda s: s.timestamp)
    energy = 0.0
    for prev, curr in zip(usable, usable[1:]):
        dt = (curr.timestamp - prev.timestamp).total_seconds()
        if dt <= 0:
            continue
        energy += (prev.watts + curr.watts) / 2.0 * dt
    return energy


@dataclass
class RunSummaryMetrics:
    energy_joules_software: float | None
    energy_per_sample_joules: float | None
    energy_per_epoch_joules: float | None
    avg_power_watts: float | None
    peak_power_watts: float | None
    samples_processed: int
    # Liczba probek mocy faktycznie uzytych do policzenia avg_power_watts/peak_power_watts
    # (po odfiltrowaniu preferred_source) - wskaznik wiarygodnosci tych dwoch metryk.
    # Przy bardzo krotkich przebiegach (np. batch_size=1) sampler_thread moze zdazyc
    # zebrac 0-1 probek w calym oknie pomiaru, przez co avg_power_watts to praktycznie
    # pojedynczy losowy odczyt zamiast realnej sredniej (patrz orchestrator.py
    # _maybe_scale_up_num_iterations, ktory temu zapobiega z wyprzedzeniem, ale ta liczba
    # zostaje zapisana zawsze, jako jawny wskaznik w dashboardzie).
    power_samples_count: int


def aggregate_run(
    power_samples: list[PowerSample],
    samples_processed: int,
    num_epochs: int | None = None,
    preferred_source: str | None = None,
) -> RunSummaryMetrics:
    """Agreguje probki mocy z jednego wybranego zrodla (preferred_source, np. 'nvml' dla GPU
    NVIDIA) w metryki run_summary. Gdy preferred_source=None, uzywa wszystkich dostepnych
    probek (przydatne, gdy jest tylko jedno aktywne zrodlo)."""
    relevant = [s for s in power_samples if s.source == preferred_source] if preferred_source else power_samples

    energy_joules = integrate_energy_joules(relevant)

    watts_values = [s.watts for s in relevant if s.watts is not None]
    avg_power = sum(watts_values) / len(watts_values) if watts_values else None
    peak_power = max(watts_values) if watts_values else None

    energy_per_sample = (
        energy_joules / samples_processed if energy_joules is not None and samples_processed > 0 else None
    )
    energy_per_epoch = energy_joules / num_epochs if energy_joules is not None and num_epochs else None

    return RunSummaryMetrics(
        energy_joules_software=energy_joules,
        energy_per_sample_joules=energy_per_sample,
        energy_per_epoch_joules=energy_per_epoch,
        avg_power_watts=avg_power,
        peak_power_watts=peak_power,
        samples_processed=samples_processed,
        power_samples_count=len(relevant),
    )


def compute_throughput(samples_processed: int, duration_s: float) -> float | None:
    if duration_s <= 0:
        return None
    return samples_processed / duration_s
