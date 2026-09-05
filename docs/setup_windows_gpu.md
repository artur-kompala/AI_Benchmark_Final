# Konfiguracja PC z GPU (Windows) — NVIDIA / AMD

Instrukcja dla maszyn testowych z dedykowanym GPU. Runner wykrywa dostępne
urządzenie automatycznie (`devices/detection.py`), ale instalacja zależności
różni się w zależności od producenta GPU, dlatego są **osobne pliki
requirements** (`requirements-nvidia.txt` / `requirements-amd.txt`) i
**osobne komendy instalacji PyTorch** (patrz niżej — nie da się tego ująć
w jednym pliku `requirements.txt`, bo torch+CUDA i torch+ROCm to różne wheele).

## NVIDIA (CUDA)

1. Zainstaluj aktualny sterownik NVIDIA (Game Ready lub Studio) — zawiera
   niezbędne biblioteki NVML używane do pomiaru mocy przez `pynvml`.
2. Utwórz i aktywuj środowisko wirtualne:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Zainstaluj PyTorch z obsługą CUDA (sprawdź aktualny numer wersji CUDA
   swojego sterownika na [pytorch.org](https://pytorch.org/get-started/locally/) —
   poniżej przykład dla CUDA 12.1):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. Zainstaluj resztę zależności:

   ```bash
   pip install -r requirements/requirements-common.txt -r requirements/requirements-nvidia.txt
   ```

5. Zweryfikuj wykrycie GPU:

   ```bash
   python scripts/check_device_availability.py
   ```

   Powinno pokazać `cuda` jako dostępne urządzenie oraz `nvml` jako
   dostępne źródło pomiaru mocy.

## AMD (ROCm)

> **Ważne ryzyko architektoniczne**: oficjalne wsparcie `torch` z backendem
> ROCm **na Windows jest ograniczone** — PyTorch/ROCm jest przede wszystkim
> testowany i wspierany na Linuksie. W praktyce może się okazać, że na
> Windows trzeba uruchomić runner w **WSL2** (Windows Subsystem for Linux)
> z dystrybucją Ubuntu, żeby uzyskać działający `torch+ROCm`. Zweryfikuj to
> jako pierwszy krok na swojej konkretnej maszynie/karcie AMD, zanim
> zaplanujesz serię pomiarową — jeśli natywny Windows nie zadziała, ta
> uwaga (wraz z wersją ROCm i kartą, która nie zadziałała) powinna trafić
> do rozdziału metodologicznego pracy jako ograniczenie badania.

### Wariant A: natywny Windows (spróbuj najpierw)

1. Zainstaluj aktualny sterownik AMD (Adrenalin).
2. Utwórz i aktywuj środowisko wirtualne (jak wyżej).
3. Spróbuj zainstalować PyTorch z obsługą ROCm zgodnie z aktualną
   instrukcją na [pytorch.org](https://pytorch.org/get-started/locally/)
   (wybierz ROCm jako compute platform). Jeśli instalator nie oferuje
   wheela ROCm dla Windows w Twojej wersji Pythona — przejdź do wariantu B.
4. Zainstaluj resztę zależności:

   ```bash
   pip install -r requirements/requirements-common.txt -r requirements/requirements-amd.txt
   ```

5. `python scripts/check_device_availability.py` — jeśli `rocm` nie jest
   wykrywany mimo zainstalowanego sterownika, sprawdź czy `rocm-smi` jest
   w PATH i czy `torch.version.hip` nie jest `None` w Twoim środowisku.

### Wariant B: WSL2 (fallback, jeśli natywny Windows nie działa)

1. Zainstaluj WSL2 z Ubuntu: `wsl --install -d Ubuntu` (PowerShell jako
   administrator), zrestartuj system jeśli wymagane.
2. Wewnątrz WSL2 zainstaluj ROCm zgodnie z oficjalną instrukcją AMD
   dla Ubuntu oraz `torch` z odpowiednim indeksem ROCm.
3. Sklonuj/skopiuj repozytorium do systemu plików WSL2 (dla wydajności
   I/O — nie operuj na `/mnt/c/...` dla ciężkich operacji na danych).
4. Reszta kroków (venv, requirements, `check_device_availability.py`)
   analogicznie jak w wariancie A, ale w terminalu WSL2.
5. **Uwaga do pomiaru mocy w WSL2**: `rocm-smi`/`pyamdgpuinfo` odczytują
   moc GPU poprawnie (dostęp do sprzętu jest przekazywany przez WSL2), ale
   próbnik `codecarbon`/RAPL dla CPU hosta może wymagać dodatkowej
   konfiguracji uprawnień — zweryfikuj `python scripts/check_device_availability.py`
   po instalacji.

## Wspólne dla obu producentów

- Warm-up (`warmup.enabled: true` w configu YAML) jest ważny szczególnie na
  laptopach/GPU z throttlingiem termicznym — pierwsze iteracje po starcie
  mają często wyższą moc/niższą wydajność niż stan ustalony.
- Jeśli chcesz wymusić konkretne urządzenie zamiast auto-detekcji, użyj
  flagi `--device cpu|cuda|rocm` przy uruchomieniu runnera.
