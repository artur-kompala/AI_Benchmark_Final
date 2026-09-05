"""Etykiety UI po polsku - terminologia spojna z rozdzialem metodologicznym pracy
(moc chwilowa, energia calkowita, energia na probke, energooszczednosc).

Jeden modul na wszystkie stringi widoczne dla uzytkownika - stad rozszerzamy TEN plik,
nie rozsypujemy tekstow po stronach."""

from __future__ import annotations

APP_TITLE = "Energooszczednosc i wydajnosc: CPU vs GPU vs NPU w zadaniach AI"

# Krotki akapit o hipotezie - uzywany na stronie 0 i (skrocony) w stopce stron.
HYPOTHESIS_SHORT = (
    "Hipoteza pracy: efektywnosc energetyczna zalezy od architektury (CPU / GPU / NPU) "
    "i parametrow wykonania. Dla modeli o umiarkowanej zlozonosci i malych rozmiarach "
    "partii (batch size) CPU moze miec porownywalna lub wyzsza efektywnosc energetyczna "
    "niz GPU, mimo nizszej maksymalnej wydajnosci. Metryka rozstrzygajaca: energia na "
    "probke [J/probke]."
)

# ---------------------------------------------------------------------------
# Nawigacja (kolejnosc stron 0-10)
# ---------------------------------------------------------------------------
NAV = {
    "start": "0. Start / Jak czytac ten dashboard",
    "efektywnosc": "1. Efektywnosc energetyczna",
    "wydajnosc": "2. Wydajnosc",
    "pareto": "3. Kompromis wydajnosc / energia (Pareto)",
    "batch": "4. Wplyw rozmiaru partii (batch size)",
    "precyzja": "5. Wplyw precyzji (fp32 / fp16 / int8)",
    "moc": "6. Moc chwilowa i temperatura",
    "para": "7. CPU vs akcelerator na tej samej maszynie",
    "pomiar": "8. Co dokladnie mierzymy / powtarzalnosc",
    "hipoteza": "9. Weryfikacja hipotezy / wnioski",
    "eksport": "10. Eksport danych",
}

# ---------------------------------------------------------------------------
# device_type - JEDEN klucz koloru/symbolu na kazdym wykresie w dashboardzie
# ---------------------------------------------------------------------------
DEVICE_TYPE_ORDER = ["cpu", "cuda", "rocm", "amd_igpu", "intel_gpu_openvino", "npu_openvino"]

DEVICE_TYPE_LABELS = {
    "cpu": "CPU",
    "cuda": "GPU NVIDIA (CUDA)",
    "rocm": "GPU AMD (ROCm)",
    "amd_igpu": "iGPU AMD",
    "intel_gpu_openvino": "iGPU Intel (OpenVINO)",
    "npu_openvino": "NPU Intel (OpenVINO)",
    "npu_directml": "NPU (DirectML)",
}

# Paleta z skilla `dataviz` (walidowana pod daltonizm) - sloty 1-6 w STALEJ kolejnosci.
# Ta sama mapa jest zrodlem prawdy dla kazdego wykresu (patrz components/theme.py).
DEVICE_TYPE_COLORS = {
    "cpu": "#2a78d6",                 # slot 1 - niebieski (protagonista hipotezy)
    "cuda": "#eb6834",                # slot 2 - pomaranczowy
    "rocm": "#1baf7a",                # slot 3 - morski
    "amd_igpu": "#eda100",            # slot 4 - zolty
    "intel_gpu_openvino": "#e87ba4",  # slot 5 - magenta
    "npu_openvino": "#008300",        # slot 6 - zielony
    "npu_directml": "#4a3aa7",        # slot 7 - fiolet (rezerwa, brak w danych)
}

# Kodowanie zlozone hue x ksztalt - potrzebne na wykresie Pareto (scatter, gdzie kazde
# dwa punkty moga sasiadowac i sama barwa moze nie wystarczyc dla >3 serii).
DEVICE_TYPE_SYMBOLS = {
    "cpu": "circle",
    "cuda": "square",
    "rocm": "diamond",
    "amd_igpu": "triangle-up",
    "intel_gpu_openvino": "cross",
    "npu_openvino": "star",
    "npu_directml": "x",
}

DEVICE_TYPE_UNKNOWN_COLOR = "#898781"  # muted gray - dla device_type spoza mapy


def device_type_label(device_type: str | None) -> str:
    return DEVICE_TYPE_LABELS.get(device_type or "", device_type or "?")


# ---------------------------------------------------------------------------
# arch_class - 4 architektury do grupowania/panelowania i tabeli decyzyjnej
# (kolor NADAL idzie po device_type - to jest tylko wymiar grupujacy)
# ---------------------------------------------------------------------------
ARCH_CPU = "CPU"
ARCH_GPU_DISCRETE = "GPU dyskretne"
ARCH_GPU_INTEGRATED = "GPU zintegrowane (iGPU)"
ARCH_NPU = "NPU (akcelerator dedykowany)"
ARCH_UNKNOWN = "GPU (kategoria nieznana)"

ARCH_CLASS_ORDER = [ARCH_CPU, ARCH_GPU_DISCRETE, ARCH_GPU_INTEGRATED, ARCH_NPU, ARCH_UNKNOWN]

# Wstecznie kompatybilne aliasy (uzywane przez starszy kod / testy)
DEVICE_CATEGORY_CPU = ARCH_CPU
DEVICE_CATEGORY_GPU_INTEGRATED = ARCH_GPU_INTEGRATED
DEVICE_CATEGORY_GPU_DISCRETE = ARCH_GPU_DISCRETE
DEVICE_CATEGORY_NPU = ARCH_NPU
DEVICE_CATEGORY_UNKNOWN_GPU = ARCH_UNKNOWN
DEVICE_CATEGORY_ORDER = ARCH_CLASS_ORDER


def device_category_label(device_type: str | None, gpu_category: str | None) -> str:
    """Klasyfikuje wiersz do jednej z czterech architektur (CPU / GPU dyskretne / GPU
    zintegrowane (iGPU) / NPU). To rozroznienie jest wymiarem grupujacym w hipotezie
    pracy (NPU = trzecia architektura obok CPU i GPU, praca par. 1.2).

    Naprawione wzgledem starej wersji:
    - `amd_igpu` -> "GPU zintegrowane (iGPU)" (wczesniej wpadalo do smieciowej kategorii),
    - `npu_openvino` -> "NPU (akcelerator dedykowany)".
    `gpu_category` pochodzi z devices.gpu_category (moze byc None dla starszych wpisow)."""
    if device_type == "cpu":
        return ARCH_CPU
    if device_type in ("npu_openvino", "npu_directml"):
        return ARCH_NPU
    if device_type == "amd_igpu":
        return ARCH_GPU_INTEGRATED
    if device_type in ("cuda", "rocm", "intel_gpu_openvino"):
        if gpu_category == "integrated":
            return ARCH_GPU_INTEGRATED
        if gpu_category == "discrete":
            return ARCH_GPU_DISCRETE
        # intel_gpu_openvino w praktyce zawsze integrated; reszta - czytelny fallback
        if device_type == "intel_gpu_openvino":
            return ARCH_GPU_INTEGRATED
        return ARCH_UNKNOWN
    return device_type or "?"


# ---------------------------------------------------------------------------
# measurement_scope - co dokladnie mierzy pomiar energii (kluczowe zastrzezenie z pracy)
# ---------------------------------------------------------------------------
SCOPE_DEVICE_ONLY = "device_only"
SCOPE_WHOLE_SYSTEM = "whole_system"
SCOPE_NPU_ONLY = "npu_only"

MEASUREMENT_SCOPE_LABELS = {
    "device_only": "tylko uklad (pomiar bezposredni CPU/GPU)",
    "npu_only": "tylko NPU",
    "whole_system": "caly SoC / komputer (pomiar posredni)",
}

SCOPE_SHORT = {
    "device_only": "tylko uklad",
    "npu_only": "tylko NPU",
    "whole_system": "caly SoC / komputer",
}

# device_type -> jaki zakres pomiaru ma w praktyce (do etykiet i szrafury serii)
DEVICE_TYPE_SCOPE = {
    "cpu": SCOPE_DEVICE_ONLY,
    "cuda": SCOPE_DEVICE_ONLY,
    "rocm": SCOPE_DEVICE_ONLY,
    "amd_igpu": SCOPE_WHOLE_SYSTEM,
    "intel_gpu_openvino": SCOPE_WHOLE_SYSTEM,
    "npu_openvino": SCOPE_WHOLE_SYSTEM,
}

SCOPE_EXPLAINER = (
    "**tylko uklad (device_only)** - pomiar programowy siega samego CPU/GPU (RAPL/zenpower "
    "dla CPU, NVML dla CUDA, ROCm-SMI dla ROCm). **caly SoC / komputer (whole_system)** - "
    "dla iGPU AMD/Intel oraz NPU nie ma licznika samego ukladu, wiec zmierzony pobor "
    "obejmuje caly pakiet SoC albo caly komputer. Watomierz Tuya (gdzie byl) mierzy pobor "
    "**calego komputera z gniazdka** - zasilacz, ekran, plyta, straty. Serii `whole_system` "
    "nie zestawiamy wprost z `device_only` bez tego zastrzezenia (oznaczone szrafura)."
)


def scope_label(scope: str | None) -> str:
    return MEASUREMENT_SCOPE_LABELS.get(scope or "", scope or "?")


# ---------------------------------------------------------------------------
# quantization_status (tylko precision=int8)
# ---------------------------------------------------------------------------
QUANTIZATION_STATUS_LABELS = {
    "success": "sukces (wszystkie kwalifikujace sie warstwy)",
    "partial": "czesciowa (np. tylko warstwy Linear)",
    "failed": "nieudana (uruchomiono fp32)",
    "": "brak / nie dotyczy",
}

# ---------------------------------------------------------------------------
# Etykiety kolumn - z jednostkami (uzywane jako tytuly osi i naglowki tabel)
# ---------------------------------------------------------------------------
COLUMN_LABELS = {
    "device_device_label": "Urzadzenie",
    "device_short": "Urzadzenie",
    "device_device_type": "Typ urzadzenia",
    "device_type": "Typ urzadzenia",
    "device_type_label": "Typ urzadzenia",
    "device_category": "Architektura",
    "arch_class": "Architektura",
    "device_vendor": "Producent",
    "device_machine_name": "Maszyna",
    "machine_name": "Maszyna",
    "task": "Zadanie",
    "model_name": "Model",
    "phase": "Faza",
    "precision": "Precyzja",
    "batch_size": "Rozmiar partii (batch size)",
    "duration_s": "Czas wykonania [s]",
    "throughput_samples_per_s": "Przepustowosc [probki/s]",
    "measurement_scope": "Zakres pomiaru mocy",
    "scope_label": "Zakres pomiaru mocy",
    "status": "Status",
    "started_at": "Data rozpoczecia",
    "repetition_group_id": "Grupa powtorzen",
    "repetition_index": "Numer powtorzenia",
    "n_repetitions": "Liczba powtorzen (n)",
    "summary_energy_joules_software": "Energia calkowita - programowo [J]",
    "summary_energy_joules_smart_plug": "Energia calkowita - watomierz [J]",
    "summary_power_discrepancy_pct": "Rozbieznosc zrodel pomiaru [%]",
    "summary_energy_per_sample_joules": "Energia na probke [J/probke]",
    "energy_per_sample_j": "Energia na probke [J/probke]",
    "energy_per_1000_samples_j": "Energia na 1000 probek [J]",
    "samples_per_joule": "Probki na dzul [probki/J]",
    "summary_energy_per_epoch_joules": "Energia na epoke [J/epoke]",
    "summary_avg_power_watts": "Srednia moc chwilowa [W]",
    "summary_peak_power_watts": "Szczytowa moc chwilowa [W]",
    "summary_flops_per_watt": "Wydajnosc obliczeniowa [FLOPS/W]",
    "summary_flops_per_sample": "FLOPS na probke",
    "summary_accuracy": "Dokladnosc (accuracy)",
    "summary_samples_processed": "Liczba przetworzonych probek",
    "summary_power_samples_count": "Liczba probek mocy",
    "summary_quantization_status": "Status kwantyzacji INT8",
    "avg_temperature_c": "Srednia temperatura [°C]",
    "mean_energy_per_sample_joules": "Srednia energia na probke [J/probke]",
    "std_energy_per_sample_joules": "Odch. std. energii na probke [J/probke]",
    "mean_duration_s": "Sredni czas wykonania [s]",
    "std_duration_s": "Odch. std. czasu wykonania [s]",
    "coefficient_of_variation_energy": "Wspolczynnik zmiennosci energii (CV)",
    "mean_energy_joules": "Srednia energia calkowita [J]",
    "std_energy_joules": "Odch. std. energii calkowitej [J]",
}

# Skrocone jednostki (do adnotacji, osi drugorzednych, tooltipow)
UNITS = {
    "energy_per_sample_j": "J/probke",
    "samples_per_joule": "probki/J",
    "throughput_samples_per_s": "probki/s",
    "duration_s": "s",
    "summary_avg_power_watts": "W",
    "summary_peak_power_watts": "W",
    "avg_temperature_c": "°C",
}

TASK_LABELS = {
    "image_classification": "Klasyfikacja obrazow",
    "nlp_sentiment": "Analiza wydzwieku (NLP)",
    "llm_inference": "Wnioskowanie LLM",
}

# precyzja = skala PORZADKOWA (fp32 > fp16 > int8 wg liczby bitow) -> jeden odcien,
# monotoniczna jasnosc (rampa niebieska z palety dataviz, kroki 700/550/400/250).
# Krok 250 to najjasniejszy dopuszczalny dla rampy porzadkowej (kontrast >=2:1 wzgledem
# tla) - walidator skilla `dataviz` (--ordinal) wylapal, ze poprzednie kroki 600/400/200/100
# schodzily za jasno (int4: 1.29:1, ponizej progu). int4 w praktyce nie wystepuje w danych
# (rezerwa), ale rampa ma byc poprawna dla wszystkich 4 zdefiniowanych stopni.
PRECISION_ORDER = ["fp32", "fp16", "int8", "int4"]
PRECISION_COLORS = {
    "fp32": "#0d366b",
    "fp16": "#1c5cab",
    "int8": "#3987e5",
    "int4": "#86b6ef",
}

PHASE_LABELS = {"train": "trening", "inference": "wnioskowanie"}


def task_label(task: str | None) -> str:
    return TASK_LABELS.get(task or "", task or "?")


def phase_label(phase: str | None) -> str:
    return PHASE_LABELS.get(phase or "", phase or "?")


def label(column: str) -> str:
    """Polska etykieta dla nazwy kolumny (z jednostka), albo sama nazwa jesli brak."""
    return COLUMN_LABELS.get(column, column)


# ---------------------------------------------------------------------------
# Naglowki bloku narracji pod kazdym wykresem
# ---------------------------------------------------------------------------
NARRATIVE_WHAT = "Co pokazuje"
NARRATIVE_HOW = "Jak czytac"
NARRATIVE_SO_WHAT = "Co z tego wynika"
NARRATIVE_HYPOTHESIS = "Odniesienie do hipotezy"

DATASET_GROWING_NOTE = (
    "Zbior danych rosnie - trwaja testy na kolejnych maszynach (m.in. *wojtek* - CPU + GPU "
    "AMD, *milosz* - CPU + GPU NVIDIA, *maciej* - CPU + NPU + iGPU). Liczby, listy maszyn i "
    "kombinacje sa wyliczane z Supabase przy kazdym odswiezeniu, wiec nowe wyniki pojawia "
    "sie tu automatycznie."
)

SMALL_N_NOTE = (
    "Slupki/punkty z n < 3 powtorzen sa pokazane z mniejszym kryciem - odchylenie "
    "standardowe z 2 powtorzen jest bardzo niepewne. n jest zawsze w tooltipie."
)
