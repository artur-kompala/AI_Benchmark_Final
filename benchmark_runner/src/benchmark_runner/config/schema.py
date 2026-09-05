"""Modele konfiguracji eksperymentu (pydantic) i rozwijanie macierzy przebiegow."""

from __future__ import annotations

import itertools
import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Precision = Literal["fp32", "fp16", "int8", "int4"]
Phase = Literal["train", "inference"]


class TrainConfig(BaseModel):
    num_epochs: int = 1
    learning_rate: float = 1e-3
    optimizer: str = "adam"


class InferenceConfig(BaseModel):
    num_iterations: int = 50


class WarmupConfig(BaseModel):
    enabled: bool = True
    iterations: int = 5


class SmartPlugConfig(BaseModel):
    # 'manual' i 'tuya' sa zaimplementowane (Faza 2 i Faza 4); tasmota/shelly/kasa to
    # miejsca zarezerwowane pod przyszle backendy (patrz power/smart_plug/).
    # Dane logowania Tuya (device_id/local_key/IP) NIE naleza tutaj - to sekrety,
    # czytane z .env (TUYA_DEVICE_ID/TUYA_LOCAL_KEY/TUYA_IP_ADDRESS), zeby nie
    # trafily do wersjonowanych plikow YAML w configs/experiments/ - patrz
    # docs/setup_smart_plug.md.
    backend: Literal["manual", "tuya", "tasmota", "shelly", "kasa"] = "manual"
    enabled: bool = True


class PowerMeterConfig(BaseModel):
    sampling_interval_ms: int = 100
    # Kolejnosc = priorytet: zenpower (realny pomiar SVI2 telemetrii VRM przez modul jadra
    # zenpower3, tylko AMD Zen, wymaga recznej instalacji - patrz docs/setup_zenpower.md)
    # > rapl (Intel, czesciowo AMD w zaleznosci od jadra/uprawnien) > codecarbon (fallback
    # estymacyjny, zawsze dostepny jesli biblioteka zainstalowana). Kolejnosc w tej liscie
    # to TYLKO punkt startowy/preferencja - _select_preferred_source w orchestrator.py i
    # tak wybiera faktycznie zrodlo z najwieksza liczba zebranych probek, wiec niedostepne
    # zrodlo na poczatku listy nigdy nie blokuje danych z kolejnego.
    cpu_sources: list[str] = Field(default_factory=lambda: ["zenpower", "rapl", "codecarbon"])
    gpu_sources: list[str] = Field(default_factory=lambda: ["nvml"])
    npu_sources: list[str] = Field(default_factory=lambda: ["npu_native"])
    # Intel iGPU (device_type="intel_gpu_openvino") - jak NPU, brak publicznego API
    # natywnego odczytu mocy, wiec zawsze fallback whole-system (patrz
    # power/intel_gpu_power_meter.py). Osobne pole od gpu_sources (nvml/rocm_smi), bo to
    # mechanistycznie ta sama sciezka co NPU, nie CUDA/ROCm.
    intel_gpu_sources: list[str] = Field(default_factory=lambda: ["intel_gpu_native"])
    # Zintegrowane AMD GPU (device_type="amd_igpu") - amdgpu_igpu_native (odczyt hwmon
    # sterownika amdgpu, patrz power/amdgpu_igpu_power_meter.py) preferowany, potem
    # fallback whole-system (bateria) jak dla NPU/Intel GPU.
    amd_igpu_sources: list[str] = Field(default_factory=lambda: ["amdgpu_igpu_native", "amd_igpu_whole_system"])
    smart_plug: SmartPlugConfig = Field(default_factory=SmartPlugConfig)


class TemperatureLoggingConfig(BaseModel):
    enabled: bool = True


class OutputConfig(BaseModel):
    save_to_supabase: bool = True
    local_buffer_path: str = "./.local_buffer/"


class LlmConfig(BaseModel):
    """Ustawienia specyficzne dla zadania 'llm_inference' - ignorowane przez inne zadania."""

    engine: Literal["onnxruntime", "llama_cpp"] = "onnxruntime"
    hf_model_name: str = "sshleifer/tiny-gpt2"
    gguf_model_path: str | None = None  # wymagane dla engine="llama_cpp"
    max_new_tokens: int = 32


class RunConfig(BaseModel):
    """Jedna konkretna kombinacja parametrow - dokladnie jeden przebieg (run).

    repetition_group_id laczy wszystkie powtorzenia (patrz ExperimentConfig.repetitions)
    tej samej kombinacji model/device/precision/batch_size/phase - uzywane do agregacji
    statystyk (run_stats_summary, patrz orchestrator._finalize_repetition_group).
    seed jest juz efektywnym seedem TEGO powtorzenia (bazowy seed + repetition_index - 1),
    nie seedem eksperymentu - powtorzenia celowo nie sa identyczne 1:1 (patrz expand_matrix)."""

    experiment_name: str
    task: str
    dataset: str | None
    model: str
    device: str  # "auto" lub konkretny device_type: cpu/cuda/rocm/npu_openvino/npu_directml
    precision: Precision
    batch_size: int
    phase: Phase
    train: TrainConfig
    inference: InferenceConfig
    warmup: WarmupConfig
    power_meter: PowerMeterConfig
    temperature_logging: TemperatureLoggingConfig
    output: OutputConfig
    llm: LlmConfig
    seed: int
    repetition_index: int = 1
    repetition_group_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ExperimentConfig(BaseModel):
    experiment_name: str
    task: str
    dataset: str | None = None
    models: list[str]
    devices: list[str] = Field(default_factory=lambda: ["auto"])
    precisions: list[Precision]
    batch_sizes: list[int]
    phases: list[Phase]
    train: TrainConfig = Field(default_factory=TrainConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    power_meter: PowerMeterConfig = Field(default_factory=PowerMeterConfig)
    temperature_logging: TemperatureLoggingConfig = Field(default_factory=TemperatureLoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    seed: int = 42
    # Liczba niezaleznych powtorzen KAZDEJ unikalnej kombinacji parametrow (model x device x
    # precision x batch_size x phase) - patrz docs/measurement_methodology.md ("Powtorzenia
    # pomiarow i statystyka"). 3 jest sensownym domyslnym kompromisem dla krotszych zadan
    # (inference/LLM); dla kosztownych kombinacji treningowych (np. ResNet-50) configi
    # jawnie obnizaja to do 2, zeby pelny sweep na wielu maszynach pozostal wykonalny w czasie.
    repetitions: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _validate_nonempty_lists(self) -> "ExperimentConfig":
        for field_name in ("models", "devices", "precisions", "batch_sizes", "phases"):
            if not getattr(self, field_name):
                raise ValueError(f"Pole '{field_name}' nie moze byc puste")
        return self

    def expand_matrix(self) -> list[RunConfig]:
        """Kartezjanski iloczyn models x devices x precisions x batch_sizes x phases, kazda
        kombinacja powielona `repetitions` razy pod rzad (ten sam repetition_group_id, unikalny
        seed = self.seed + (repetition_index - 1) per powtorzenie - patrz RunConfig).

        KAZDY RunConfig dostaje WLASNA, glebok kopie zagniezdzonych podconfigow (train/
        inference/warmup/power_meter/temperature_logging/output/llm) - przekazanie tego
        samego obiektu self.train/self.inference/... (bez .model_copy) do wielu RunConfig
        sprawia, ze pydantic uzywa TEJ SAMEJ instancji przez referencje we wszystkich,
        wiec mutacja w jednym przebiegu (np. core/orchestrator.py::_maybe_scale_up_
        num_iterations podnoszace run_spec.inference.num_iterations dla konkretnego
        batch_size) natychmiast "przecieka" do WSZYSTKICH innych RunConfig w tej samej
        macierzy, w tym o zupelnie innym batch_size (potwierdzony bug: identyczne
        przeskalowane num_iterations dla kazdego batch_size, mimo drastycznie roznego
        czasu iteracji)."""
        combos = itertools.product(self.models, self.devices, self.precisions, self.batch_sizes, self.phases)
        run_configs: list[RunConfig] = []
        for model, device, precision, batch_size, phase in combos:
            repetition_group_id = str(uuid.uuid4())
            for repetition_index in range(1, self.repetitions + 1):
                run_configs.append(
                    RunConfig(
                        experiment_name=self.experiment_name,
                        task=self.task,
                        dataset=self.dataset,
                        model=model,
                        device=device,
                        precision=precision,
                        batch_size=batch_size,
                        phase=phase,
                        train=self.train.model_copy(deep=True),
                        inference=self.inference.model_copy(deep=True),
                        warmup=self.warmup.model_copy(deep=True),
                        power_meter=self.power_meter.model_copy(deep=True),
                        temperature_logging=self.temperature_logging.model_copy(deep=True),
                        output=self.output.model_copy(deep=True),
                        llm=self.llm.model_copy(deep=True),
                        seed=self.seed + (repetition_index - 1),
                        repetition_index=repetition_index,
                        repetition_group_id=repetition_group_id,
                    )
                )
        return run_configs
