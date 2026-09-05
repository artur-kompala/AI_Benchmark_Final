"""Diagnostyka: wypisuje wykryte urzadzenia i dostepne zrodla pomiaru mocy na tej maszynie.

Uzycie:
    python scripts/check_device_availability.py

Wymaga aktywnego venv z zainstalowanymi zaleznosciami wlasciwymi dla tej maszyny
(patrz README.md). Przydatne jako pierwszy krok po skonfigurowaniu nowej maszyny testowej.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from benchmark_runner.devices.detection import detect_available_devices  # noqa: E402
from benchmark_runner.power.amdgpu_igpu_power_meter import AmdIgpuPowerSampler, AmdIgpuWholeSystemPowerSampler  # noqa: E402
from benchmark_runner.power.codecarbon_meter import CodecarbonPowerSampler  # noqa: E402
from benchmark_runner.power.intel_gpu_power_meter import IntelGpuPowerSampler  # noqa: E402
from benchmark_runner.power.npu_power_meter import NpuPowerSampler  # noqa: E402
from benchmark_runner.power.nvml_meter import NvmlPowerSampler  # noqa: E402
from benchmark_runner.power.rapl_meter import RaplPowerSampler  # noqa: E402
from benchmark_runner.power.rocm_smi_meter import RocmSmiPowerSampler  # noqa: E402
from benchmark_runner.power.smart_plug.tuya_reader import TuyaConnection, TuyaPowerSampler, load_tuya_config_from_env  # noqa: E402
from benchmark_runner.power.zenpower_meter import ZenpowerPowerSampler  # noqa: E402
from benchmark_runner.utils.system_info import collect_system_info  # noqa: E402

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def main() -> None:
    load_dotenv(dotenv_path=_ENV_PATH)

    print("=== System ===")
    info = collect_system_info()
    print(f"  Maszyna: {info.machine_name}")
    print(f"  OS: {info.os_name} {info.os_version}")
    print(f"  CPU: {info.cpu_model}")
    print(f"  RAM: {info.total_ram_gb} GB" if info.total_ram_gb else "  RAM: nieznana")

    print("\n=== Wykryte urzadzenia obliczeniowe ===")
    devices = detect_available_devices()
    for device in devices:
        print(f"  - {device.device_type}: {device.label} (backend={device.backend}, driver={device.driver_version})")
    if not devices:
        print("  (brak - to nie powinno sie zdarzyc, CPU zawsze powinien byc dostepny)")

    print("\n=== Dostepne zrodla pomiaru mocy ===")
    checks = [
        ("zenpower (CPU, AMD Zen)", ZenpowerPowerSampler()),
        ("rapl (CPU)", RaplPowerSampler()),
        ("codecarbon (CPU)", CodecarbonPowerSampler()),
        ("nvml (NVIDIA GPU)", NvmlPowerSampler()),
        ("rocm_smi (AMD GPU dyskretne)", RocmSmiPowerSampler()),
        ("amdgpu_igpu_native (AMD GPU zintegrowane)", AmdIgpuPowerSampler()),
        ("amd_igpu_whole_system (AMD iGPU fallback)", AmdIgpuWholeSystemPowerSampler()),
        ("npu_native/whole-system (NPU fallback)", NpuPowerSampler()),
        ("intel_gpu_native/whole-system (Intel GPU fallback)", IntelGpuPowerSampler()),
    ]
    for label, sampler in checks:
        try:
            available = sampler.is_available()
        except Exception as exc:
            print(f"  - {label}: BLAD przy sprawdzaniu ({exc})")
            continue
        status = "DOSTEPNE" if available else "niedostepne"
        print(f"  - {label}: {status}")
        try:
            sampler.close()
        except Exception:
            pass

    print("\n=== Watomierz fizyczny ===")
    tuya_config = load_tuya_config_from_env()
    if tuya_config is None:
        print("  - tuya: niedostepne (TUYA_DEVICE_ID/TUYA_LOCAL_KEY/TUYA_IP_ADDRESS puste w .env)")
        print("    Bez tych danych runner uzywa trybu manualnego - patrz docs/setup_smart_plug.md.")
    else:
        sampler = TuyaPowerSampler(TuyaConnection(tuya_config))
        available = sampler.is_available()
        print(f"  - tuya ({tuya_config.ip_address}): {'DOSTEPNE' if available else 'BLAD POLACZENIA'}")
        if not available:
            print("    Sprawdz IP/local_key/wersje protokolu w .env, patrz docs/setup_smart_plug.md.")

    print(
        "\nUwaga: pomiar mocy NPU, Intel GPU (intel_gpu_openvino) i AMD iGPU fallback "
        "(amd_igpu_whole_system) dziala tylko na baterii (bez zasilacza) - patrz "
        "docs/setup_npu_intel.md, docs/setup_amd_igpu.md i docs/measurement_methodology.md."
    )
    print(
        "Uwaga: zenpower (moc CPU AMD Zen) wymaga recznej instalacji modulu jadra "
        "zenpower3 przez DKMS - opcjonalne, patrz docs/setup_zenpower.md. Bez niego "
        "runner automatycznie uzywa rapl/codecarbon."
    )


if __name__ == "__main__":
    main()
