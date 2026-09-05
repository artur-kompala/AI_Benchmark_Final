"""Rdzen runnera: przeplyw pojedynczego przebiegu (run), od configu do zapisu w Supabase.

Kolejnosc dla kazdego RunConfig z rozwinietej macierzy eksperymentu:
resolve device -> task.prepare() -> warmup -> start smart_plug + power sampler thread ->
wykonanie treningu/inferencji (perf_counter) -> stop sampler -> end smart_plug ->
agregacja metryk + discrepancy -> zapis do Supabase (lub local buffer przy braku sieci) ->
teardown. Blad pojedynczego przebiegu jest logowany i NIE przerywa reszty serii.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from benchmark_runner.config.loader import load_config
from benchmark_runner.config.schema import ExperimentConfig, PowerMeterConfig, RunConfig, SmartPlugConfig
from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.core.base_task import TaskStepResult
from benchmark_runner.core.registry import get_task_class
from benchmark_runner.core.run_context import RunContext
from benchmark_runner.devices.detection import resolve_device
from benchmark_runner.metrics.aggregation import aggregate_run, compute_throughput, integrate_energy_joules
from benchmark_runner.metrics.discrepancy import compute_discrepancy, wh_to_joules
from benchmark_runner.power.amdgpu_igpu_power_meter import AmdIgpuPowerSampler, AmdIgpuWholeSystemPowerSampler
from benchmark_runner.power.base_power_meter import MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE, BasePowerSampler, PowerSample
from benchmark_runner.power.codecarbon_meter import CodecarbonPowerSampler
from benchmark_runner.power.intel_gpu_power_meter import IntelGpuPowerSampler
from benchmark_runner.power.npu_power_meter import NpuPowerSampler
from benchmark_runner.power.nvml_meter import NvmlPowerSampler
from benchmark_runner.power.rapl_meter import RaplPowerSampler
from benchmark_runner.power.rocm_smi_meter import RocmSmiPowerSampler
from benchmark_runner.power.sampler_thread import PowerSamplingThread
from benchmark_runner.power.zenpower_meter import ZenpowerPowerSampler
from benchmark_runner.power.smart_plug.base import SmartPlugReader
from benchmark_runner.power.smart_plug.manual_reader import ManualSmartPlugReader
from benchmark_runner.power.smart_plug.tuya_reader import (
    TuyaConnection,
    TuyaPowerSampler,
    TuyaSmartPlugReader,
    load_tuya_config_from_env,
)
from benchmark_runner.power.temperature import read_temperature
from benchmark_runner.storage.local_buffer import LocalBuffer
from benchmark_runner.storage.models import (
    DeviceRecord,
    PowerSampleRecord,
    RunRecord,
    RunStatsSummaryRecord,
    RunSummaryRecord,
)
from benchmark_runner.storage.repository import RepositoryError, RunRepository
from benchmark_runner.utils.hashing import hash_config
from benchmark_runner.utils.system_info import SystemInfo, collect_system_info

# Import rejestrujacy zadania przez dekorator @register_task w TASK_REGISTRY.
from benchmark_runner.tasks.image_classification import task as _image_classification_task  # noqa: F401
from benchmark_runner.tasks.llm_inference import task as _llm_inference_task  # noqa: F401
from benchmark_runner.tasks.nlp_sentiment import task as _nlp_sentiment_task  # noqa: F401

logger = logging.getLogger(__name__)

_SOURCE_SAMPLER_MAP: dict[str, dict[str, type[BasePowerSampler]]] = {
    "cpu": {"zenpower": ZenpowerPowerSampler, "rapl": RaplPowerSampler, "codecarbon": CodecarbonPowerSampler},
    "cuda": {"nvml": NvmlPowerSampler},
    "rocm": {"rocm_smi": RocmSmiPowerSampler},
    "npu_openvino": {"npu_native": NpuPowerSampler},
    "npu_directml": {"npu_native": NpuPowerSampler},
    "intel_gpu_openvino": {"intel_gpu_native": IntelGpuPowerSampler},
    "amd_igpu": {
        "amdgpu_igpu_native": AmdIgpuPowerSampler,
        "amd_igpu_whole_system": AmdIgpuWholeSystemPowerSampler,
    },
}

# Urzadzenia bez publicznego API natywnego odczytu mocy - zawsze fallback whole-system
# (tempo rozladowania baterii, dziala tylko na baterii) - patrz measurement_scope nizej
# i docs/measurement_methodology.md. 'amd_igpu' jest tu KONSERWATYWNIE, mimo ze
# AmdIgpuPowerSampler (hwmon sterownika amdgpu) moze w praktyce dawac odczyt blizszy
# device_only - bez weryfikacji na prawdziwym sprzecie, czy ta wartosc faktycznie izoluje
# sam blok graficzny APU (a nie caly SoC), nie da sie tego uczciwie oznaczyc inaczej.
_WHOLE_SYSTEM_ONLY_DEVICE_TYPES = ("npu_openvino", "npu_directml", "intel_gpu_openvino", "amd_igpu")


def _requested_sources(device_type: str, power_meter: PowerMeterConfig) -> list[str]:
    if device_type == "cpu":
        return power_meter.cpu_sources
    if device_type in ("cuda", "rocm"):
        return power_meter.gpu_sources
    if device_type in ("npu_openvino", "npu_directml"):
        return power_meter.npu_sources
    if device_type == "intel_gpu_openvino":
        return power_meter.intel_gpu_sources
    if device_type == "amd_igpu":
        return power_meter.amd_igpu_sources
    return []


def _select_preferred_source(requested: list[str], power_samples: list[PowerSample]) -> str | None:
    """Wybiera zrodlo z NAJWIEKSZA liczba faktycznie zebranych probek (watts != None)
    sposrod skonfigurowanych `requested` (np. power_meter.cpu_sources=['rapl',
    'codecarbon']) - zamiast sztywno pierwszego z listy niezaleznie od dostepnosci.

    Blad: dla device_type='cpu' preferowanym zrodlem bylo zawsze requested[0]='rapl',
    ale rapl bywa niedostepny na danej maszynie (0 probek), podczas gdy codecarbon
    (requested[1]) dziala poprawnie i zbiera setki probek - mimo to avg_power_watts/
    energy_joules_software zapisywaly sie jako None (bo aggregate_run filtrowal probki
    po pustym 'rapl'), CICHO, bez zadnego bledu/wyjatku sygnalizujacego problem.

    Przy remisie (rowna liczba probek, w tym 0=0 gdy nic nie zebralo danych) zachowuje
    kolejnosc z configu - max() z Pythona zwraca pierwszy element o maksymalnej wartosci
    klucza, wiec requested[0] wygrywa remis, deterministycznie i przewidywalnie."""
    if not requested:
        return None
    counts = dict.fromkeys(requested, 0)
    for sample in power_samples:
        if sample.source in counts and sample.watts is not None:
            counts[sample.source] += 1
    return max(requested, key=lambda source: counts[source])


def _check_power_plausibility(device_type: str, avg_power_watts: float | None) -> str | None:
    """Sprawdza, czy avg_power_watts miesci sie w fizycznie prawdopodobnym zakresie dla
    device_type (patrz power/base_power_meter.py::MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE).
    Zwraca czytelny komunikat (do error_message/logu), gdy przekroczono prog, albo None
    gdy wszystko w porzadku / brak progu dla tego device_type / avg_power_watts=None.

    DRUGA linia obrony - RaplPowerSampler/CodecarbonPowerSampler juz odrzucaja pojedyncze
    implauzybilne probki U ZRODLA (patrz ich docstringi - blad: przy zablokowanym
    uprawnieniami RAPL, PermissionError na /sys/class/powercap/intel-rapl/.../energy_uj,
    codecarbon potrafil wpasc w niestabilny stan i zwrocic 300-1050W dla laptopa), ale ta
    funkcja lapie przypadek, gdyby mimo to SREDNIA wyszla nieprawdopodobna (np. kilka
    osobno "akceptowalnych" probek dajacych nieprawdopodobna srednia), plus dziala jako
    zabezpieczenie dla przyszlych zrodel bez wlasnej walidacji."""
    if avg_power_watts is None:
        return None
    threshold = MAX_PLAUSIBLE_WATTS_BY_DEVICE_TYPE.get(device_type)
    if threshold is None or avg_power_watts <= threshold:
        return None
    return (
        f"avg_power_watts={avg_power_watts:.1f}W przekracza prog fizycznej wiarygodnosci dla "
        f"device_type='{device_type}' ({threshold:.0f}W) - fizycznie nieprawdopodobne dla tego "
        f"urzadzenia, dane mocy odrzucone (prawdopodobny blad odczytu zrodla pomiaru mocy)"
    )


def build_power_samplers(device: DeviceProfile, power_meter: PowerMeterConfig) -> list[BasePowerSampler]:
    sampler_classes = _SOURCE_SAMPLER_MAP.get(device.device_type, {})
    samplers: list[BasePowerSampler] = []
    for source_name in _requested_sources(device.device_type, power_meter):
        sampler_cls = sampler_classes.get(source_name)
        if sampler_cls is None:
            logger.warning(
                "Nieznane/niewspierane zrodlo pomiaru mocy '%s' dla urzadzenia '%s' - pomijam",
                source_name,
                device.device_type,
            )
            continue
        samplers.append(sampler_cls())
    return samplers


def build_smart_plug(config: SmartPlugConfig) -> tuple[SmartPlugReader, BasePowerSampler | None]:
    """Zwraca (SmartPlugReader do odczytu energii skumulowanej na start/koniec runu,
    opcjonalny BasePowerSampler do ciaglego probkowania mocy chwilowej - None gdy
    backend tego nie wspiera, np. tryb manualny)."""
    if not config.enabled:
        # config.enabled=False (recznie w YAML, albo --no-smart-plug) musi wylaczyc TAKZE
        # ciagle probkowanie (TuyaPowerSampler), nie tylko SmartPlugReader.start_run/end_run
        # - w przeciwnym razie proba polaczenia z (mozliwe ze juz nieosiagalna) wtyczka
        # nadal by sie odbywala w tle przy kazdym przebiegu, mimo jawnego wylaczenia.
        return ManualSmartPlugReader(enabled=False), None
    if config.backend == "tuya":
        tuya_config = load_tuya_config_from_env()
        if tuya_config is not None:
            connection = TuyaConnection(tuya_config)
            return TuyaSmartPlugReader(connection, enabled=config.enabled), TuyaPowerSampler(connection)
        # load_tuya_config_from_env() juz zalogowalo ostrzezenie - graceful fallback ponizej.
    elif config.backend != "manual":
        logger.warning(
            "Backend watomierza '%s' nie jest jeszcze zaimplementowany - uzywam trybu manualnego",
            config.backend,
        )
    return ManualSmartPlugReader(enabled=config.enabled), None


def _seed_everything(seed: int) -> None:
    """Ustawia seed dla random/numpy/torch na poczatku kazdego przebiegu - dzieki temu
    powtorzenia (repetitions) z roznym seedem (patrz RunConfig.seed) faktycznie daja rozne
    losowe losowanie batcha/inicjalizacji wag zamiast identycznego przebiegu 1:1."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


# Ponizej próg (w liczbie probek mocy w calym oknie pomiaru) uznajemy przebieg za zbyt
# krotki dla wiarygodnej avg_power_watts - potwierdzone empirycznie: przy batch_size=1
# (duration_s~0.14-0.15s) sampler_thread z domyslnym sampling_interval_ms=100 zdazyl
# zebrac 0-1 probek, a 3 powtorzenia tej samej konfiguracji dawaly 17.6W/7.3W/15.4W -
# rozrzut 2-3x, bo avg_power_watts to byl praktycznie pojedynczy losowy odczyt.
_MIN_POWER_SAMPLES_THRESHOLD = 20
# Cel skalowania - lekko powyzej progu, zeby nie balansowac dokladnie na granicy przy
# niewielkich wahaniach czasu iteracji miedzy warmup a wlasciwym pomiarem.
_MIN_POWER_SAMPLES_TARGET = 25
# Twardy limit skalowania w gore - CZASOWY (sekundy skumulowanego pomiaru), nie
# wielokrotnosc num_iterations z configu. Blad poprzedniej wersji (limit = configured *
# 20x): dla wysokiej przepustowosci (duzy batch_size) nawet 20x moze byc krotsze niz
# jednorazowy narzut startowy niektorych zrodel mocy (np. codecarbon: is_available()~1.4s
# + pierwszy sample()~3.1s, zmierzone empirycznie) - potwierdzone: 1000 iteracji (=20x
# configu=50) przy batch_size=64 zajmowalo tylko ~1.6-2.7s, wiec limit byl osiagany PRZED
# tym, jak sampler zdazyl oddac pierwsza probke, niezaleznie od tego jak bardzo
# rzeczywistosc "chciala" wiecej iteracji. Limit czasowy odzwierciedla rzeczywiste
# ograniczenie (narzut startowy zrodla + docelowa liczba probek), nie arbitralna
# wielokrotnosc liczby iteracji z configu.
_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S = 20.0


def _maybe_scale_up_num_iterations(run_spec: RunConfig, warmup_duration_s: float) -> None:
    """PIERWSZE, best-effort oszacowanie liczby iteracji na podstawie czasu warmup - TYLKO
    punkt startowy, zeby ograniczyc liczbe prob w retry loop nizej (_run_single), NIE
    gwarancja osiagniecia progu. Empirycznie potwierdzone: warmup.iterations bywa male
    (dry-run: 1, domyslnie: 5), a pierwsze wywolanie po kompilacji OpenVINO/ONNX Runtime
    bywa nieproporcjonalnie wolniejsze od kolejnych (jednorazowy koszt "rozgrzania" cache
    kerneli w samym silniku wykonawczym, nie zwiazany z eksportem/kompilacja modelu, ktore
    dzieja sie wczesniej w prepare() i sa juz policzone PRZED pomiarem warmup) - usredniony
    per_iter_s z tak malej proby systematycznie ZAWYZA prawdziwy koszt iteracji w stanie
    ustalonym, przez co ta funkcja NIEDOSZACOWUJE potrzebnej liczby iteracji (potwierdzone:
    po przeskalowaniu do 171 iteracji faktyczna liczba probek mocy wynosila 4-14, nie ~25).
    Autorytatywna korekta na podstawie NAPRAWDE zmierzonego czasu dzieje sie w retry loop w
    _run_single (patrz _num_additional_iterations_needed), ktory dziala na rzeczywistych
    danych z tego samego przebiegu zamiast na proxy z warmup."""
    if run_spec.warmup.iterations <= 0 or warmup_duration_s <= 0:
        return

    per_iter_s = warmup_duration_s / run_spec.warmup.iterations
    sampling_interval_s = run_spec.power_meter.sampling_interval_ms / 1000.0
    if sampling_interval_s <= 0:
        return

    configured_iterations = run_spec.inference.num_iterations
    estimated_duration_s = per_iter_s * configured_iterations
    expected_samples = estimated_duration_s / sampling_interval_s
    if expected_samples >= _MIN_POWER_SAMPLES_THRESHOLD:
        return

    target_iterations = math.ceil(_MIN_POWER_SAMPLES_TARGET * sampling_interval_s / per_iter_s)
    max_allowed = math.floor(_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S / per_iter_s)
    new_num_iterations = min(target_iterations, max_allowed)
    if new_num_iterations <= configured_iterations:
        return

    logger.info(
        "Przebieg zbyt krotki dla wiarygodnego pomiaru mocy (~%.1f probek przy "
        "num_iterations=%d, sampling_interval_ms=%d) - podnosze num_iterations %d -> %d "
        "dla TEGO przebiegu (model=%s, device=%s, batch_size=%d), zeby zebrac >=%d probek mocy",
        expected_samples,
        configured_iterations,
        run_spec.power_meter.sampling_interval_ms,
        configured_iterations,
        new_num_iterations,
        run_spec.model,
        run_spec.device,
        run_spec.batch_size,
        _MIN_POWER_SAMPLES_TARGET,
    )
    run_spec.inference.num_iterations = new_num_iterations


# Maksymalna liczba prob pomiaru w retry loop (_run_single) - pierwsza proba + korekty na
# podstawie realnie zmierzonego czasu. Ograniczone, zeby nigdy nie zawiesic serii w
# nieskonczonosc, jesli z jakiegos powodu (np. skrajnie zmienny czas iteracji) proba za
# proba nie osiaga progu - wtedy przebieg zapisuje sie z tym, co realnie udalo sie zebrac,
# z ostrzezeniem w logu, zamiast blokowac cala serie.
_MAX_POWER_SAMPLING_ATTEMPTS = 3


def _num_additional_iterations_needed(
    *,
    collected_samples: int,
    cumulative_duration_s: float,
    latest_attempt_duration_s: float,
    latest_attempt_iterations: int,
    sampling_interval_s: float,
    target_samples: int,
    max_cumulative_duration_s: float,
) -> int | None:
    """Liczy o ile iteracji wydluzyc KOLEJNA probe (dodatek DO used_iterations, nie nowa
    wartosc od zera), zeby SKUMULOWANY czas pomiaru (cumulative_duration_s, suma wszystkich
    dotychczasowych prob TEGO przebiegu) wystarczyl na zebranie target_samples probek.

    Blad naprawiony tutaj: poprzednia wersja liczyla "ile iteracji potrzebowalaby JEDNA,
    IZOLOWANA proba, zeby SAMA osiagnela target_samples" - przy stalym per-iteration time
    zadania ta wartosc jest STALA niezaleznie od numeru proby, wiec po jednym podniesieniu
    do tej wartosci kolejne wywolanie sugerowalo DOKLADNIE TO SAMO num_iterations -> retry
    loop uznawal to za "brak postepu" i przerywal, MIMO ZE realnie zebrana liczba probek
    (sample_count()) nadal wynosila 0. Przyczyna: sampler (np. codecarbon) ma jednorazowy,
    stosunkowo duzy narzut startowy (potwierdzone empirycznie: is_available()=~1.4s +
    PIERWSZY sample()/flush()=~3.1s, ZANIM zacznie faktycznie probkowac w tempie
    sampling_interval_ms), ktory NIE powtarza sie miedzy probami (sampler nie jest
    restartowany - patrz _run_single), ale POCHLANIA znaczna czesc wczesnych prob, zanim
    jakiekolwiek probki zaczna splywac. Formula ponizej rozroznia dwa przypadki:

    - collected_samples > 0: mamy juz obserwowana, REALNA szybkosc probkowania
      (collected_samples / cumulative_duration_s) z calego dotychczasowego,
      nieprzerwanego okna - ekstrapolujemy wprost ile jeszcze CZASU (nie: ile "nowych
      iteracji od zera") potrzeba na brakujace probki.
    - collected_samples == 0: nie mamy jeszcze zadnej obserwowanej szybkosci (caly
      dotychczasowy czas zjadl jednorazowy narzut startowy) - zamiast poddawac sie,
      podwajamy dotychczasowy SKUMULOWANY czas pomiaru jako nastepny krok (z minimum
      sampling_interval_s * target_samples), zeby dac samplerowi kolejna, wieksza szanse
      "dogonic" ten narzut i zaczac faktycznie probkowac.

    max_cumulative_duration_s ogranicza SUME dotychczasowego + dodatkowego czasu (nie
    liczbe iteracji vs. configu) - zabezpiecza przed patologicznym rozdmuchaniem przebiegu
    w nieskonczonosc, jednoczesnie dopuszczajac tyle iteracji, ile faktycznie potrzeba przy
    wysokiej przepustowosci (duzy batch_size), gdzie nawet duza wielokrotnosc num_iterations
    z configu moze nadal trwac krocej niz jednorazowy narzut startowy zrodla.

    Zwraca liczbe DODATKOWYCH iteracji (>=0) albo None, gdy nie da sie nic sensownie
    policzyc (latest_attempt_iterations<=0, latest_attempt_duration_s<=0)."""
    if latest_attempt_iterations <= 0 or latest_attempt_duration_s <= 0 or sampling_interval_s <= 0:
        return None
    per_iter_s = latest_attempt_duration_s / latest_attempt_iterations
    if per_iter_s <= 0:
        return None

    remaining_budget_s = max_cumulative_duration_s - cumulative_duration_s
    if remaining_budget_s <= 0:
        return 0

    if collected_samples > 0 and cumulative_duration_s > 0:
        observed_rate = collected_samples / cumulative_duration_s  # probek/s, z calego okna
        remaining_samples = target_samples - collected_samples
        additional_time_s = remaining_samples / observed_rate if observed_rate > 0 else cumulative_duration_s
    else:
        additional_time_s = max(cumulative_duration_s, sampling_interval_s * target_samples)

    additional_time_s = min(additional_time_s, remaining_budget_s)
    return math.ceil(additional_time_s / per_iter_s)


# Kwantyzacja INT8 w tym projekcie dotyczy TYLKO fazy inference (patrz utils/quantization.py) -
# kombinacje precision=int8 + phase=train sa pomijane PRZED probą wykonania (zamiast konczyc sie
# bledem w trakcie), z czytelnym logiem zamiast wpisu w runs.status='failed'.
_UNSUPPORTED_PRECISION_PHASE = {("int8", "train")}

# batch_size=1 + phase=train dla image_classification zawsze konczy sie bledem PyTorch
# ("Expected more than 1 value per channel when training") - MobileNetV3/ResNet-50 uzywaja
# BatchNorm, ktory podczas treningu liczy statystyki PO BATCHU i wymaga batch_size>1 (w
# odroznieniu od inferencji, gdzie BatchNorm uzywa juz zapisanych statystyk running_mean/var
# i batch_size=1 dziala normalnie - stad ten wyjatek dotyczy TYLKO fazy train). nlp_sentiment
# (DistilBERT, LayerNorm) nie ma tego ograniczenia - normalizacja liczona per-probka, nie
# per-batch - stad filtr jest zawezony do task=='image_classification', nie ogolny.
_UNSUPPORTED_TASK_BATCH_SIZE_PHASE = {("image_classification", 1, "train")}


def _filter_unsupported_combos(run_specs: list[RunConfig]) -> list[RunConfig]:
    kept: list[RunConfig] = []
    for run_spec in run_specs:
        if (run_spec.precision, run_spec.phase) in _UNSUPPORTED_PRECISION_PHASE:
            logger.info(
                "Pomijam kombinacje precision=%s + phase=%s (model=%s, device=%s, batch_size=%d) - "
                "w tym projekcie nie trenuje sie w INT8, tylko inferencja",
                run_spec.precision,
                run_spec.phase,
                run_spec.model,
                run_spec.device,
                run_spec.batch_size,
            )
            continue
        if (run_spec.task, run_spec.batch_size, run_spec.phase) in _UNSUPPORTED_TASK_BATCH_SIZE_PHASE:
            logger.info(
                "Pomijam kombinacje task=%s + batch_size=%d + phase=%s (model=%s, device=%s, "
                "precision=%s) - BatchNorm nie wspiera treningu z batch_size=1 ('Expected more "
                "than 1 value per channel when training')",
                run_spec.task,
                run_spec.batch_size,
                run_spec.phase,
                run_spec.model,
                run_spec.device,
                run_spec.precision,
            )
            continue
        kept.append(run_spec)
    return kept


# Sanity check PRZED zapisem do Supabase: accuracy bliskie poziomowi przypadku zdradza
# model o niewczytanych/losowych wagach - typowo sciezke OpenVINO/ONNX Runtime na
# NPU/Intel GPU bez wczytanego checkpointu referencyjnego (patrz
# tasks/image_classification/onnx_export.py::load_reference_checkpoint). CIFAR-10 ma
# 10 klas -> poziom przypadku ~0.10, prog 0.3 jest istotnie wyzszy, ale wciaz ponizej
# realistycznej accuracy nawet dla mocno skwantyzowanego/niedotrenowanego modelu.
_MIN_PLAUSIBLE_ACCURACY_BY_TASK: dict[str, float] = {
    "image_classification": 0.3,
}


class ImplausibleAccuracyError(RuntimeError):
    """Rzucany, gdy accuracy z inferencji jest podejrzanie blisko poziomu przypadku.
    Rzucany WEWNATRZ _run_single's try/except, wiec przebieg konczy sie normalnie jako
    status='failed' z czytelnym error_message (ta sama sciezka co kazdy inny wyjatek w
    trakcie przebiegu) - dane nadal trafiaja do Supabase (do wgladu/diagnozy), ale NIE
    jako 'completed', wiec nie zanieczyszczaja analizy."""


def _check_accuracy_sanity(run_spec: RunConfig, result: TaskStepResult) -> None:
    if run_spec.phase != "inference" or result.accuracy is None:
        return
    threshold = _MIN_PLAUSIBLE_ACCURACY_BY_TASK.get(run_spec.task)
    if threshold is None or result.accuracy >= threshold:
        return
    raise ImplausibleAccuracyError(
        f"accuracy={result.accuracy:.4f} jest ponizej progu wiarygodnosci ({threshold}) dla "
        f"zadania '{run_spec.task}' (model={run_spec.model}, device={run_spec.device}, "
        f"precision={run_spec.precision}) - wyglada na model o niewytrenowanych/losowych "
        f"wagach zamiast wytrenowanego checkpointu. Przebieg oznaczony jako status='failed', "
        f"zeby nie zanieczyscic analizy podejrzanym wynikiem."
    )


@dataclass
class _RunOutcome:
    """Wynik jednego przebiegu potrzebny do agregacji statystyk grupy powtorzen
    (patrz ExperimentRunner._finalize_repetition_group) - None dla pol niedostepnych
    (np. przebieg 'failed', albo zapis trafil do lokalnego bufora bez znanego device_id)."""

    status: str
    device_id: str | None
    duration_s: float | None
    energy_joules_software: float | None


def _apply_dry_run_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """--dry-run: male liczby epok/iteracji, zeby szybko zweryfikowac ze caly przeplyw
    (device -> pomiar -> zapis w Supabase) dziala bez wyjatkow, bez pelnego treningu."""
    config.warmup.iterations = min(config.warmup.iterations, 1)
    config.train.num_epochs = 1
    config.inference.num_iterations = min(config.inference.num_iterations, 3)
    logger.info(
        "Tryb --dry-run: warmup=%d iter, train.num_epochs=1, inference.num_iterations=%d",
        config.warmup.iterations,
        config.inference.num_iterations,
    )
    return config


class ExperimentRunner:
    def __init__(self, repository: RunRepository, local_buffer: LocalBuffer) -> None:
        self._repository = repository
        self._local_buffer = local_buffer
        self._system_info: SystemInfo = collect_system_info()
        # Dlugozyjace watki probkujace moc device-side (rapl/codecarbon/nvml/rocm_smi/
        # npu_native/intel_gpu_native), keyowane po device_type - tworzone RAZ na cala
        # serie eksperymentow (patrz _get_persistent_sampling_thread), nie per-przebieg.
        # Blad naprawiony tym mechanizmem: kazdy przebieg budowal te zrodla od zera,
        # placac pelny jednorazowy narzut startowy (zmierzone empirycznie na tym projekcie:
        # CodecarbonPowerSampler.is_available()~1.4s + pierwszy sample()~3.1s) przy KAZDYM
        # z dziesiatkow/setek przebiegow w tej samej serii, zamiast raz.
        self._persistent_power_threads: dict[str, PowerSamplingThread] = {}

    def _get_persistent_sampling_thread(self, device: DeviceProfile, power_meter: PowerMeterConfig) -> PowerSamplingThread:
        """Zwraca dlugozyjacy PowerSamplingThread dla device.device_type, tworzac go przy
        pierwszym uzyciu i reuzywajac dla wszystkich kolejnych przebiegow tego samego typu
        urzadzenia w tej samej serii. Zaklada, ze power_meter jest jednolite w calej serii
        (jedno pole ExperimentConfig.power_meter, kopiowane .model_copy(deep=True) do
        kazdego RunConfig - patrz config/schema.py::expand_matrix) - sampling_interval_ms
        z PIERWSZEGO uzycia dla danego device_type obowiazuje przez reszte serii."""
        existing = self._persistent_power_threads.get(device.device_type)
        if existing is not None:
            return existing
        power_samplers = build_power_samplers(device, power_meter)
        thread = PowerSamplingThread(power_samplers, interval_ms=power_meter.sampling_interval_ms).start()
        self._persistent_power_threads[device.device_type] = thread
        logger.info(
            "Zainicjalizowano dlugozyjace zrodla pomiaru mocy dla device_type='%s' (%d zrodel "
            "skonfigurowanych) - beda reuzywane (bez ponownej inicjalizacji) dla wszystkich "
            "przebiegow tego typu urzadzenia w tej serii",
            device.device_type,
            len(power_samplers),
        )
        return thread

    def _close_persistent_sampling_threads(self) -> None:
        """Zamyka wszystkie dlugozyjace watki probkujace moc na koniec calej serii -
        kazdy .stop() zamyka tez swoje zrodla (np. EmissionsTracker.stop() dla
        codecarbon). Wolane w finally w run_from_config, wiec dzieje sie nawet gdy seria
        zostanie przerwana wyjatkiem w trakcie."""
        for device_type, thread in self._persistent_power_threads.items():
            try:
                thread.stop()
            except Exception:
                logger.warning(
                    "Blad przy zamykaniu dlugozyjacego watku probkujacego mocy dla '%s'", device_type, exc_info=True
                )
        self._persistent_power_threads.clear()

    def run_from_config(
        self,
        config_path: str,
        device_override: str | None = None,
        dry_run: bool = False,
        no_smart_plug: bool = False,
    ) -> None:
        experiment_config = load_config(config_path)
        if dry_run:
            experiment_config = _apply_dry_run_overrides(experiment_config)
        if no_smart_plug:
            # Nadpisuje config YAML niezaleznie od backend/enabled - patrz --no-smart-plug
            # w cli.py po uzasadnienie (unika blokowania sesji bez nadzoru w trybie manual).
            experiment_config.power_meter.smart_plug.enabled = False
            logger.info("--no-smart-plug: watomierz fizyczny wylaczony dla calej serii")
        run_specs = _filter_unsupported_combos(experiment_config.expand_matrix())
        logger.info(
            "Wczytano config '%s' - %d przebiegow do wykonania (repetitions=%d)",
            experiment_config.experiment_name,
            len(run_specs),
            experiment_config.repetitions,
        )

        group_id: str | None = None
        group_meta: RunConfig | None = None
        group_outcomes: list[_RunOutcome] = []

        try:
            for i, run_spec in enumerate(run_specs, start=1):
                logger.info(
                    "=== Przebieg %d/%d: model=%s device=%s precision=%s batch_size=%d phase=%s "
                    "(powtorzenie %d/%d, grupa=%s) ===",
                    i,
                    len(run_specs),
                    run_spec.model,
                    device_override or run_spec.device,
                    run_spec.precision,
                    run_spec.batch_size,
                    run_spec.phase,
                    run_spec.repetition_index,
                    experiment_config.repetitions,
                    run_spec.repetition_group_id,
                )

                if group_id is not None and run_spec.repetition_group_id != group_id:
                    self._finalize_repetition_group(group_meta, group_outcomes)
                    group_outcomes = []

                group_id = run_spec.repetition_group_id
                group_meta = run_spec

                try:
                    outcome = self._run_single(run_spec, device_override)
                except Exception:
                    logger.error("Przebieg %d/%d przerwany nieoczekiwanym bledem - kontynuuje serie", i, len(run_specs), exc_info=True)
                    outcome = None
                if outcome is not None:
                    group_outcomes.append(outcome)

            if group_meta is not None:
                self._finalize_repetition_group(group_meta, group_outcomes)
        finally:
            # Zamyka dlugozyjace zrodla pomiaru mocy (patrz _get_persistent_sampling_thread)
            # na koniec calej serii, niezaleznie od tego czy zakonczyla sie normalnie czy
            # zostala przerwana wyjatkiem/Ctrl+C w trakcie.
            self._close_persistent_sampling_threads()

        flushed = self._local_buffer.flush_pending(self._repository)
        if flushed:
            logger.info("Na koniec sesji wyslano %d wczesniej zbuforowanych przebiegow", flushed)

    def _finalize_repetition_group(self, group_meta: RunConfig, outcomes: list[_RunOutcome]) -> None:
        """Liczy mean/std/wspolczynnik zmiennosci po WSZYSTKICH powtorzeniach jednej
        kombinacji parametrow i zapisuje do run_stats_summary. Uwzglednia tylko przebiegi
        'completed' z poprawnie ustalonym device_id (przebiegi zbuforowane lokalnie z powodu
        braku sieci nie maja jeszcze device_id z Supabase w momencie liczenia statystyk -
        znana, udokumentowana tutaj luka: ich flush pod koniec sesji NIE doksztaltuje juz
        zapisanej wczesniej statystyki grupy)."""
        usable = [o for o in outcomes if o.status == "completed" and o.device_id is not None]
        if not usable:
            logger.info(
                "Grupa powtorzen %s (model=%s, device=%s, precision=%s, batch_size=%d, phase=%s) "
                "bez ani jednego udanego przebiegu - pomijam run_stats_summary",
                group_meta.repetition_group_id,
                group_meta.model,
                group_meta.device,
                group_meta.precision,
                group_meta.batch_size,
                group_meta.phase,
            )
            return

        durations = [o.duration_s for o in usable if o.duration_s is not None]
        energies = [o.energy_joules_software for o in usable if o.energy_joules_software is not None]

        mean_duration = statistics.mean(durations) if durations else None
        std_duration = statistics.stdev(durations) if len(durations) >= 2 else (0.0 if durations else None)
        mean_energy = statistics.mean(energies) if energies else None
        std_energy = statistics.stdev(energies) if len(energies) >= 2 else (0.0 if energies else None)
        cv_energy = std_energy / mean_energy if mean_energy not in (None, 0) and std_energy is not None else None

        stats_record = RunStatsSummaryRecord(
            repetition_group_id=group_meta.repetition_group_id,
            task=group_meta.task,
            model_name=group_meta.model,
            device_id=usable[0].device_id,
            precision=group_meta.precision,
            batch_size=group_meta.batch_size,
            phase=group_meta.phase,
            n_repetitions=len(usable),
            mean_duration_s=mean_duration,
            std_duration_s=std_duration,
            mean_energy_joules=mean_energy,
            std_energy_joules=std_energy,
            coefficient_of_variation_energy=cv_energy,
        )
        try:
            self._repository.upsert_run_stats_summary(stats_record)
        except RepositoryError as exc:
            logger.warning(
                "Nie udalo sie zapisac run_stats_summary dla grupy %s - statystyka pominieta "
                "(surowe przebiegi w 'runs' juz sa zapisane, mozna doliczyc pozniej z SQL). Przyczyna: %s",
                group_meta.repetition_group_id,
                exc,
            )

    def _run_single(self, run_spec: RunConfig, device_override: str | None) -> _RunOutcome:
        _seed_everything(run_spec.seed)
        device = resolve_device(device_override or run_spec.device)

        run_context = RunContext(run_spec=run_spec, device=device, system_info=self._system_info)
        task = get_task_class(run_spec.task)(run_context)

        status = "completed"
        error_message: str | None = None
        samples_processed = 0
        power_samples: list[PowerSample] = []
        smart_plug_reading = None
        sampling_thread: PowerSamplingThread | None = None
        smart_plug_thread: PowerSamplingThread | None = None
        smart_plug: SmartPlugReader | None = None
        duration_s = 0.0
        started_at = finished_at = datetime.now(timezone.utc)
        result = None
        flops_per_sample: float | None = None

        try:
            task.prepare()

            try:
                flops_per_sample = task.estimate_flops_per_sample()
            except Exception:
                # Best-effort, patrz BaseTask.estimate_flops_per_sample()/metrics/flops.py -
                # niepowodzenie oszacowania FLOPS nie powinno nigdy zepsuc calego przebiegu.
                logger.warning("Nie udalo sie oszacowac FLOPS dla przebiegu %s", run_context.run_id, exc_info=True)

            if run_spec.warmup.enabled:
                t_warmup0 = time.perf_counter()
                task.warmup(run_spec.warmup.iterations)
                warmup_duration_s = time.perf_counter() - t_warmup0
                if run_spec.phase == "inference":
                    _maybe_scale_up_num_iterations(run_spec, warmup_duration_s)

            # Dlugozyjacy sampler device-side (rapl/codecarbon/nvml/rocm_smi/npu_native/
            # intel_gpu_native), wspoldzielony miedzy WSZYSTKIMI przebiegami tego samego
            # device_type w tej serii (patrz _get_persistent_sampling_thread) - drain()
            # czysci probki z przerwy miedzy przebiegami / z prepare()+warmup TEGO
            # przebiegu, ale watek zyje dalej (nie stop()) dla kolejnych przebiegow.
            sampling_thread = self._get_persistent_sampling_thread(device, run_spec.power_meter)
            sampling_thread.drain()

            # Watomierz fizyczny (jesli wlaczony) NIE jest wspoldzielony miedzy przebiegami -
            # SmartPlugReader.start_run/end_run musi byc wolane per run_id (mierzy energie
            # skumulowana W TYM oknie), wiec zostaje na osobnym, krotkotrwalym watku - w
            # odroznieniu od dlugozyjacego zrodla device-side wyzej.
            smart_plug, smart_plug_sampler = build_smart_plug(run_spec.power_meter.smart_plug)
            smart_plug.start_run(run_context.run_id)
            if smart_plug_sampler is not None:
                smart_plug_thread = PowerSamplingThread(
                    [smart_plug_sampler], interval_ms=run_spec.power_meter.sampling_interval_ms
                ).start()

            # Retry loop (TYLKO faza inference): _maybe_scale_up_num_iterations wyzej to
            # tylko oszacowanie z warmup, empirycznie niewystarczajace (systematycznie
            # zawyza per-iteration time, wiec NIEDOSZACOWUJE potrzebnej liczby iteracji -
            # potwierdzone: po przeskalowaniu do 171 iteracji faktyczna liczba probek mocy
            # wynosila 4-14, nie ~25). Tutaj po KAZDEJ probie sprawdzamy naprawde zebrana
            # liczbe probek (sampling_thread.sample_count(), bez zatrzymywania watku - wiec
            # kolejna proba kontynuuje TO SAMO okno pomiarowe, nie zaczyna nowego) i jesli
            # nadal ponizej progu, doliczamy iteracje na podstawie REALNEGO zmierzonego
            # czasu tej konkretnej proby. samples_processed/duration_s sumuja sie po
            # wszystkich probach - to jest spojne z power_samples, ktore tez pochodza z
            # calego, nieprzerwanego okna obejmujacego wszystkie proby.
            started_at = datetime.now(timezone.utc)
            sampling_interval_s = run_spec.power_meter.sampling_interval_ms / 1000.0

            for attempt in range(1, _MAX_POWER_SAMPLING_ATTEMPTS + 1):
                used_iterations = run_spec.inference.num_iterations
                t0 = time.perf_counter()
                result = task.run_train_epoch() if run_spec.phase == "train" else task.run_inference_pass()
                attempt_duration_s = time.perf_counter() - t0

                duration_s += attempt_duration_s
                samples_processed += result.samples_processed

                if run_spec.phase != "inference":
                    break  # retry dotyczy tylko inferencji - epoka treningowa nie jest tu powtarzana

                collected = sampling_thread.sample_count()
                if collected >= _MIN_POWER_SAMPLES_THRESHOLD:
                    break
                if attempt == _MAX_POWER_SAMPLING_ATTEMPTS:
                    logger.warning(
                        "Po %d probach nadal tylko %d probek mocy (prog=%d) dla przebiegu %s "
                        "(model=%s, device=%s, batch_size=%d) - zapisuje z tym, co udalo sie "
                        "zebrac, zamiast blokowac serie w nieskonczonosc",
                        attempt,
                        collected,
                        _MIN_POWER_SAMPLES_THRESHOLD,
                        run_context.run_id,
                        run_spec.model,
                        run_spec.device,
                        run_spec.batch_size,
                    )
                    break

                # Uzywa SKUMULOWANEGO duration_s (suma wszystkich dotychczasowych prob),
                # NIE tylko attempt_duration_s tej jednej proby - inaczej formula liczy
                # "ile iteracji potrzebowalaby JEDNA, IZOLOWANA proba", co przy stalym
                # per-iteration time daje ZAWSZE ten sam wynik i po jednym kroku wyglada
                # jak "brak postepu", mimo ze sampler realnie wciaz nie zdazyl zebrac zadnej
                # probki (jednorazowy narzut startowy, patrz docstring funkcji nizej).
                additional_iterations = _num_additional_iterations_needed(
                    collected_samples=collected,
                    cumulative_duration_s=duration_s,
                    latest_attempt_duration_s=attempt_duration_s,
                    latest_attempt_iterations=used_iterations,
                    sampling_interval_s=sampling_interval_s,
                    target_samples=_MIN_POWER_SAMPLES_TARGET,
                    max_cumulative_duration_s=_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S,
                )
                new_num_iterations = used_iterations + additional_iterations if additional_iterations else used_iterations
                if new_num_iterations <= used_iterations:
                    # Osiagnieto limit czasowy calkowitego pomiaru
                    # (_MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S) - kolejna proba i tak nic by
                    # nie zmieniła.
                    logger.warning(
                        "Proba %d/%d dla przebiegu %s: %d probek mocy (prog=%d), ale osiagnieto "
                        "limit czasowy calkowitego pomiaru (%.1fs) - przerywam proby, zapisuje "
                        "z tym co jest",
                        attempt,
                        _MAX_POWER_SAMPLING_ATTEMPTS,
                        run_context.run_id,
                        collected,
                        _MIN_POWER_SAMPLES_THRESHOLD,
                        _MAX_POWER_SAMPLING_CUMULATIVE_DURATION_S,
                    )
                    break

                logger.info(
                    "Proba %d/%d dla przebiegu %s: %d probek mocy (prog=%d) po %.4fs skumulowanego "
                    "pomiaru (NIE oszacowanie z warmup) - podnosze num_iterations %d -> %d "
                    "(model=%s, device=%s, batch_size=%d)",
                    attempt,
                    _MAX_POWER_SAMPLING_ATTEMPTS,
                    run_context.run_id,
                    collected,
                    _MIN_POWER_SAMPLES_THRESHOLD,
                    duration_s,
                    used_iterations,
                    new_num_iterations,
                    run_spec.model,
                    run_spec.device,
                    run_spec.batch_size,
                )
                run_spec.inference.num_iterations = new_num_iterations

            finished_at = datetime.now(timezone.utc)
            _check_accuracy_sanity(run_spec, result)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            finished_at = datetime.now(timezone.utc)
            logger.error("Blad podczas wykonywania przebiegu %s", run_context.run_id, exc_info=True)
        finally:
            # sampling_thread.drain() (NIE .stop()) - dlugozyjacy watek zyje dalej dla
            # kolejnych przebiegow tego samego device_type (patrz _get_persistent_
            # sampling_thread); smart_plug_thread NATOMIAST jest per-przebieg wiec .stop()
            # jak dawniej.
            if sampling_thread is not None:
                power_samples = sampling_thread.drain()
            if smart_plug_thread is not None:
                power_samples = power_samples + smart_plug_thread.stop()
            if smart_plug is not None:
                smart_plug_reading = smart_plug.end_run(run_context.run_id)
            try:
                task.teardown()
            except Exception:
                logger.warning("Blad podczas teardown zadania %s", run_context.run_id, exc_info=True)

        quantization_status = result.extra.get("quantization_status") if result is not None else None
        accuracy = result.accuracy if result is not None else None

        return self._finalize_and_persist(
            run_spec=run_spec,
            device=device,
            status=status,
            error_message=error_message,
            samples_processed=samples_processed,
            power_samples=power_samples,
            smart_plug_reading=smart_plug_reading,
            duration_s=duration_s,
            started_at=started_at,
            finished_at=finished_at,
            quantization_status=quantization_status,
            accuracy=accuracy,
            flops_per_sample=flops_per_sample,
        )

    def _finalize_and_persist(
        self,
        *,
        run_spec: RunConfig,
        device: DeviceProfile,
        status: str,
        error_message: str | None,
        samples_processed: int,
        power_samples: list[PowerSample],
        smart_plug_reading,
        duration_s: float,
        started_at: datetime,
        finished_at: datetime,
        quantization_status: str | None = None,
        accuracy: float | None = None,
        flops_per_sample: float | None = None,
    ) -> _RunOutcome:
        measurement_scope = "whole_system" if device.device_type in _WHOLE_SYSTEM_ONLY_DEVICE_TYPES else "device_only"

        requested = _requested_sources(device.device_type, run_spec.power_meter)
        preferred_source = _select_preferred_source(requested, power_samples)

        summary_metrics = aggregate_run(
            power_samples,
            samples_processed=samples_processed,
            num_epochs=run_spec.train.num_epochs if run_spec.phase == "train" else None,
            preferred_source=preferred_source,
        )
        throughput = compute_throughput(samples_processed, duration_s)

        # Diagnostyka: dokladnie KTORE zrodlo dostarczylo probki uzyte do energy_joules_
        # software/avg_power_watts, jawnie w logu - bez tego trzeba by zgadywac z samej
        # koncowej wartosci liczbowej, czy pochodzi z oczekiwanego zrodla (np.
        # intel_gpu_native), czy z czegos innego/wcale (0 probek -> avg_power_watts=None).
        if power_samples:
            source_counts: dict[str, int] = {}
            for sample in power_samples:
                source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
        else:
            source_counts = {}

        # Surowy rozklad probek uzytych do avg_power_watts (nie tylko finalna srednia) -
        # min/median obok juz istniejacych avg_power_watts/peak_power_watts (=max), zeby
        # dalo sie odroznic w logu "jedna probka-odstajaca zanizyla srednia" od "caly
        # przebieg realnie mial nizszy pobor mocy" (co mogloby wskazywac na throttling/
        # DVFS drivera GPU obnizajacy czestotliwosc w niektorych przebiegach). Ten sam
        # filtr po preferred_source co wewnatrz aggregate_run(), zeby min/median dotyczyly
        # DOKLADNIE tych samych probek co avg_power_watts/peak_power_watts.
        relevant_samples = (
            [s for s in power_samples if s.source == preferred_source] if preferred_source else power_samples
        )
        relevant_watts = [s.watts for s in relevant_samples if s.watts is not None]
        min_power_watts = min(relevant_watts) if relevant_watts else None
        median_power_watts = statistics.median(relevant_watts) if relevant_watts else None

        logger.info(
            "Przebieg (model=%s, device=%s, precision=%s, batch_size=%s, phase=%s): zrodlo "
            "preferowane dla energy_joules_software/avg_power_watts = '%s' (%d probek z tego "
            "zrodla / %d probek total ze wszystkich zrodel: %s) - rozklad mocy: "
            "min=%s, median=%s, avg=%s, max=%s W",
            run_spec.model,
            device.device_type,
            run_spec.precision,
            run_spec.batch_size,
            run_spec.phase,
            preferred_source,
            summary_metrics.power_samples_count,
            len(power_samples),
            source_counts or "brak probek z zadnego zrodla",
            f"{min_power_watts:.2f}" if min_power_watts is not None else None,
            f"{median_power_watts:.2f}" if median_power_watts is not None else None,
            f"{summary_metrics.avg_power_watts:.2f}" if summary_metrics.avg_power_watts is not None else None,
            f"{summary_metrics.peak_power_watts:.2f}" if summary_metrics.peak_power_watts is not None else None,
        )

        power_implausibility_reason = _check_power_plausibility(device.device_type, summary_metrics.avg_power_watts)
        if power_implausibility_reason is not None:
            logger.warning(
                "Przebieg (model=%s, device=%s, precision=%s, batch_size=%s, phase=%s): %s - "
                "zerowanie pol mocy (avg_power_watts/peak_power_watts/energy_joules_software/"
                "energy_per_sample_joules/energy_per_epoch_joules) i oznaczanie przebiegu jako "
                "status='partial' zamiast cichego zapisania fizycznie niemozliwej wartosci",
                run_spec.model,
                device.device_type,
                run_spec.precision,
                run_spec.batch_size,
                run_spec.phase,
                power_implausibility_reason,
            )
            summary_metrics.avg_power_watts = None
            summary_metrics.peak_power_watts = None
            summary_metrics.energy_joules_software = None
            summary_metrics.energy_per_sample_joules = None
            summary_metrics.energy_per_epoch_joules = None
            if status == "completed":
                status = "partial"
            if error_message is None:
                error_message = power_implausibility_reason

        # FLOPS/W = flops_per_sample / avg_power_watts (patrz docs/measurement_methodology.md) -
        # FLOPS modelu szacowane statycznie (metrics/flops.py), niezaleznie od pomiaru energii
        # w tym konkretnym przebiegu; None gdy ktoras skladowa niedostepna (np. task nie
        # wspiera estymacji FLOPS, albo brak aktywnego zrodla pomiaru mocy).
        flops_per_watt = (
            flops_per_sample / summary_metrics.avg_power_watts
            if flops_per_sample is not None and summary_metrics.avg_power_watts
            else None
        )

        energy_smart_plug_j = wh_to_joules(smart_plug_reading.energy_wh) if smart_plug_reading else None
        if energy_smart_plug_j is None:
            # Fallback: niektore wtyczki Tuya nie udostepniaja przez DPS wlasnego
            # skumulowanego licznika energii (np. ATORCH S1BW - patrz
            # docs/setup_smart_plug.md), wiec SmartPlugReader.end_run() zwraca None.
            # Jesli mamy ciagle probki mocy z tego samego urzadzenia
            # (TuyaPowerSampler, source='smart_plug'), oszacuj energie przez ich
            # calkowanie zamiast zostawiac pole puste.
            smart_plug_power_samples = [s for s in power_samples if s.source == "smart_plug"]
            energy_smart_plug_j = integrate_energy_joules(smart_plug_power_samples)
        discrepancy = compute_discrepancy(summary_metrics.energy_joules_software, energy_smart_plug_j)

        avg_temperature = read_temperature(device)

        config_snapshot = run_spec.model_dump(mode="json")
        config_hash = hash_config(config_snapshot)

        device_record = DeviceRecord(
            machine_name=self._system_info.machine_name,
            device_type=device.device_type,
            device_label=device.label,
            vendor=device.vendor,
            cpu_model=self._system_info.cpu_model,
            gpu_model=device.label if device.device_type != "cpu" else None,
            os_name=self._system_info.os_name,
            os_version=self._system_info.os_version,
            driver_version=device.driver_version,
            total_ram_gb=self._system_info.total_ram_gb,
            gpu_category=device.gpu_category,
        )

        run_record = RunRecord(
            device_id="",  # uzupelniane przy zapisie, po get_or_create_device
            task=run_spec.task,
            model_name=run_spec.model,
            phase=run_spec.phase,
            dataset=run_spec.dataset,
            batch_size=run_spec.batch_size,
            precision=run_spec.precision,
            num_epochs=run_spec.train.num_epochs if run_spec.phase == "train" else None,
            num_iterations=run_spec.inference.num_iterations if run_spec.phase == "inference" else None,
            config_hash=config_hash,
            config_snapshot=config_snapshot,
            measurement_scope=measurement_scope,
            warmup_done=run_spec.warmup.enabled,
            status=status,
            error_message=error_message,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            throughput_samples_per_s=throughput,
            avg_temperature_c=avg_temperature,
            repetition_group_id=run_spec.repetition_group_id,
            repetition_index=run_spec.repetition_index,
        )

        power_sample_records = [
            PowerSampleRecord(run_id="", timestamp=s.timestamp, source=s.source, watts=s.watts, temperature_c=s.temperature_c)
            for s in power_samples
        ]
        has_real_smart_plug_samples = any(s.source == "smart_plug" for s in power_samples)
        if smart_plug_reading is not None and not has_real_smart_plug_samples:
            # Watomierz bez ciaglego probkowania (np. tryb manualny) - dopisz pojedynczy
            # rekord-znacznik bez wartosci mocy, zeby bylo widac ze watomierz zostal
            # uzyty w tym przebiegu. Backendy z ciaglym probkowaniem (np. Tuya) maja juz
            # prawdziwe probki w power_samples powyzej - znacznik byłby tylko szumem.
            power_sample_records.append(
                PowerSampleRecord(run_id="", timestamp=finished_at, source="smart_plug", watts=None)
            )

        summary_record = RunSummaryRecord(
            run_id="",
            energy_joules_software=summary_metrics.energy_joules_software,
            energy_joules_smart_plug=energy_smart_plug_j,
            power_discrepancy_pct=discrepancy,
            energy_per_sample_joules=summary_metrics.energy_per_sample_joules,
            energy_per_epoch_joules=summary_metrics.energy_per_epoch_joules,
            avg_power_watts=summary_metrics.avg_power_watts,
            peak_power_watts=summary_metrics.peak_power_watts,
            flops_per_sample=flops_per_sample,
            flops_per_watt=flops_per_watt,
            samples_processed=samples_processed,
            power_samples_count=summary_metrics.power_samples_count,
            quantization_status=quantization_status,
            accuracy=accuracy,
        )

        device_id = self._persist(device_record, run_record, power_sample_records, summary_record)
        return _RunOutcome(
            status=status,
            device_id=device_id,
            duration_s=duration_s if status == "completed" else None,
            energy_joules_software=summary_metrics.energy_joules_software,
        )

    def _persist(
        self,
        device_record: DeviceRecord,
        run_record: RunRecord,
        power_sample_records: list[PowerSampleRecord],
        summary_record: RunSummaryRecord,
    ) -> str | None:
        """Zwraca device_id gdy zapis do Supabase sie udal, None gdy trafil do lokalnego
        bufora (device_id z Supabase wtedy jeszcze nieznany - patrz _finalize_repetition_group)."""
        try:
            device_id = self._repository.get_or_create_device(device_record)
            run_record.device_id = device_id
            run_id = self._repository.insert_run(run_record)
            for ps in power_sample_records:
                ps.run_id = run_id
            self._repository.insert_power_samples_batch(power_sample_records)
            summary_record.run_id = run_id
            self._repository.insert_run_summary(summary_record)
            return device_id
        except RepositoryError as exc:
            logger.warning(
                "Nie udalo sie zapisac wyniku do Supabase - buforuje lokalnie do ponownej proby. Przyczyna: %s",
                exc,
            )
            self._local_buffer.enqueue_run_bundle(device_record, run_record, power_sample_records, summary_record)
            return None
