"""Detekcja zintegrowanego GPU AMD (iGPU, np. Radeon Vega w APU Ryzen 5000H, Radeon 780M
w APU Ryzen 7040/8040) przez PyTorch skompilowany z ROCm (torch.version.hip) - jak
devices/rocm.py, ale filtruje TYLKO urzadzenia sklasyfikowane jako 'integrated' (patrz
gpu_classification.py), zeby (na hipotetycznej maszynie z jednoczesnie dyskretna i
zintegrowana karta AMD) nie kolidowac z devices/rocm.py (dyskretne karty). Na wszystkich
znanych dzis maszynach testowych to rozroznienie i tak nigdy nie koliduje: maszyny z
dyskretna karta AMD (Artur PC) nie maja iGPU (desktop), a maszyny z iGPU AMD (Ola,
Proksza) nie maja dyskretnej karty AMD (ich dyskretne GPU to NVIDIA, obslugiwane osobno
przez devices/cuda.py).

WAZNE OGRANICZENIE (niezweryfikowane na prawdziwym sprzecie w trakcie implementacji):
ROCm historycznie ma niepewne/niekompletne wsparcie dla zintegrowanych GPU (APU) - w
zaleznosci od wersji ROCm i architektury karty, torch.cuda.is_available() moze zwracac
False dla iGPU, ktory faktycznie jest obecny w systemie, dopoki nie zostanie ustawiona
zmienna srodowiskowa HSA_OVERRIDE_GFX_VERSION (typowy workaround dla architektur bez
oficjalnego wsparcia ROCm, np. gfx90c dla Vega w Ryzen 5000H, gfx1103 dla 780M) - patrz
docs/setup_amd_igpu.md po instrukcje i znane zastrzezenia. Ten modul CELOWO nie probuje
zgadywac/ustawiac tej zmiennej sam (wplyw na caly proces Pythona, w tym inne urzadzenia) -
uzytkownik ustawia ja przed uruchomieniem runnera, jesli jest potrzebna na jego sprzecie.
"""

from __future__ import annotations

import logging

from benchmark_runner.core.base_device import DeviceProfile
from benchmark_runner.devices.gpu_classification import classify_gpu_category

logger = logging.getLogger(__name__)


def _find_integrated_amd_device_index() -> int | None:
    try:
        import torch

        if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
            return None
        for index in range(torch.cuda.device_count()):
            label = torch.cuda.get_device_name(index)
            if classify_gpu_category("amd", label) == "integrated":
                return index
        return None
    except Exception:
        logger.debug("PyTorch/ROCm niedostepne na tej maszynie (amd_igpu)", exc_info=True)
        return None


def is_available() -> bool:
    return _find_integrated_amd_device_index() is not None


def get_profile() -> DeviceProfile:
    import torch

    index = _find_integrated_amd_device_index()
    if index is None:
        raise RuntimeError("Zintegrowane AMD GPU niedostepne na tej maszynie (torch+ROCm go nie widzi)")
    label = torch.cuda.get_device_name(index)
    return DeviceProfile(
        device_type="amd_igpu",
        label=label,
        vendor="amd",
        torch_device_str=f"cuda:{index}",  # PyTorch mapuje HIP (ROCm) na to samo API co CUDA
        backend="pytorch",
        driver_version=torch.version.hip,
        gpu_category="integrated",
    )
