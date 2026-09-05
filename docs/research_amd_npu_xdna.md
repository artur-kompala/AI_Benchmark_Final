# Research spike: wsparcie AMD NPU (Ryzen AI / XDNA) — Proksza

> **To jest raport/rekomendacja, NIE implementacja.** Zgodnie z zadaniem, `configs/
> experiments/NPU/` i kod orchestratora **nie zostały zmodyfikowane**. Poniżej podsumowanie
> rozpoznania (przeprowadzonego zdalnie, przez research — **bez fizycznego dostępu do
> Proksza** w trakcie pisania tego dokumentu, patrz zastrzeżenia przy każdym punkcie) i
> rekomendacja co dalej.

## Kontekst

Ryzen AI (XDNA) to zupełnie inny stos niż OpenVINO NPU (Intel), używany dziś w tym
projekcie dla `npu_openvino` (patrz `docs/setup_npu_intel.md`). Nie ma gotowego OpenVINO
NPU plugin dla AMD — trzeba by użyć zupełnie innej ścieżki technologicznej.

## Pytanie 1: Czy `amdxdna` jest dostępny w kernelu na Prokszy?

**Nie zweryfikowane bezpośrednio — brak dostępu do Proksza w trakcie tego rozpoznania.**
Poniżej stan ogólny sterownika (aktualny na dzień pisania, wrzesień 2026), i dokładna
komenda do samodzielnego sprawdzenia na Prokszy.

Sterownik jądra `amdxdna` jest **upstreamowany do głównej gałęzi Linuksa od wersji 6.14**
(początek 2025) i od tego czasu aktywnie rozwijany — trwają prace nad kolejnymi
usprawnieniami (m.in. patche eksponujące metryki poboru mocy NPU, poprawki firmware,
nowe ioctl) kierowane do kolejnych wydań, włącznie z oknem scalania 7.1
([Phoronix](https://www.phoronix.com/news/Ryzen-AI-NPU-Linux-Power-Metric),
[Phoronix Forums](https://www.phoronix.com/forums/forum/linux-graphics-x-org-drivers/open-source-amd-linux/1509807-amd-npu-firmware-upstreamed-for-the-ryzen-ai-amdxdna-driver-coming-in-linux-6-14),
[dokumentacja jądra](https://docs.kernel.org/accel/amdxdna/amdnpu.html)).

**Sprawdź na Prokszy:**

```bash
uname -r                    # wersja jadra - potrzeba >=6.14 dla wbudowanego wsparcia
modinfo amdxdna             # czy modul jest w ogole znany temu jadru
lsmod | grep amdxdna        # czy jest zaladowany
ls /sys/class/accel/         # amdxdna rejestruje sie jako "accel" device
```

Jeśli `modinfo amdxdna` nic nie zwraca mimo jądra ≥6.14, dystrybucja mogła wyłączyć ten
moduł w konfiguracji jądra — wtedy potrzebna byłaby własna kompilacja jądra (realistycznie
poza budżetem tego zadania, patrz rekomendacja niżej).

## Pytanie 2: Czy XRT + ONNX Runtime Vitis AI EP da się zainstalować bez konfliktu z
istniejącymi zależnościami?

Dwie odrębne ścieżki, obie **poza standardowym `pip install`** do istniejącego venv:

### Ścieżka A: Oficjalny stos AMD (Ryzen AI Software + Vitis AI Execution Provider)

**Stan zmienił się istotnie w trakcie 2026** — starsza dokumentacja ONNX Runtime
(`onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html`) nadal mówi
wprost: *"Ryzen AI Linux support is not enabled in the Vitis AI ONNX Runtime Execution
Provider release"*. Ale **nowsze ogłoszenia AMD z 2026 (CES 2026, Ryzen AI Software
1.6.1 i 1.8.0 z sierpnia 2026) wprowadziły "early access" wsparcie Linux**:

- Tylko **Ubuntu 24.04 LTS** jest oficjalnie wspierany.
- Dostęp jest **ograniczony do zarejestrowanych klientów AMD** ("early access") — czas
  oczekiwania na zatwierdzenie rejestracji jest nieznany i poza naszą kontrolą.
- Wspierane formaty: modele CNN w INT8 lub BF16 (MobileNetV3 to CNN — pasuje), modele NLP
  w BF16, LLM tylko dla wykonania na NPU (zwykle wymaga 64GB+ RAM systemowego).

([Phoronix](https://www.phoronix.com/news/Ryzen-AI-Software-1.6.1),
[Tom's Hardware](https://www.tomshardware.com/news/amd-asks-do-you-need-ryzen-ai-support-in-linux),
[AMD CES 2026](https://www.amd.com/en/newsroom/press-releases/2026-1-5-amd-expands-ai-leadership-across-client-graphics-.html),
[Ryzen AI Release 1.8.0](https://ryzenai.docs.amd.com/_/downloads/en/latest/pdf/))

**Ryzyko konfliktu zależności**: Ryzen AI Software instaluje **własny build** ONNX
Runtime z wbudowanym Vitis AI EP — realistycznie wymaga dedykowanego venv, osobnego od
istniejącego `requirements-npu.txt` (który zakłada `onnxruntime`/`onnxruntime-directml`
generyczne, dla ścieżki DirectML/Windows) — analogicznie do problemu znalezionego w
Zadaniu 2 (osobny venv `torch+ROCm` niekompatybilny z `torch+CUDA` w tym samym
środowisku).

### Ścieżka B: Community (IRON / MLIR-AIE / Peano)

Aktywnie rozwijany, otwarty stos ([IRON/MLIR-AIE](https://xilinx.github.io/mlir-aie/)) —
**"open MLIR-AIE / IRON kernel stack runs transformer and conv models on the NPU under
Linux, with a host-CPU fallback for ops that are not yet on-device"**. Nie wymaga
rejestracji u AMD, działa na już-upstreamowanym `amdxdna` + XRT. Przepływ: model → IRON
(kompilator Python) → MLIR-AIE → Peano (LLVM dla AI Engine) → XRT ładuje na NPU.

Istnieją też mniejsze, eksperymentalne projekty społecznościowe (np.
[`open-xdna`](https://github.com/Scottcjn/open-xdna) — "verified matmul on first-gen
XDNA1", [`xdna-engine`](https://github.com/atassis/xdna-engine) dla XDNA2/Strix) — na tak
wczesnym etapie rozwoju, że nie nadają się jeszcze jako podstawa pod pełny pipeline
benchmarku.

**To NIE jest `pip install` biblioteki** — to własny, niskopoziomowy toolchain
kompilatorowy (LLVM-owy backend dla architektury AI Engine), wymagający budowy z
źródeł i pisania/kompilowania kerneli dla docelowego modelu, nie automatycznego
importu gotowego modelu PyTorch/ONNX.

## Pytanie 3: Realistyczny szacunek nakładu pracy na jeden przebieg inferencji MobileNetV3

**Ścieżka A (oficjalna)**: nawet przy pomyślnym zatwierdzeniu early access i zgodnej
wersji Ubuntu, integracja z tym projektem wymagałaby nowego backendu (analogicznego do
`openvino`/`onnxruntime` w `tasks/image_classification/task.py` — export do formatu
akceptowanego przez Vitis AI EP, obsługa wymogu INT8/BF16, nowy `devices/amd_npu.py`,
nowy `power/` sampler dla NPU AMD — prawdopodobnie znów whole-system, bo brak
udokumentowanego per-device API mocy). Szacunek: **2-5 dni pracy integracyjnej PO
otrzymaniu dostępu early access** — sam czas oczekiwania na zatwierdzenie rejestracji
jest dodatkowym, nieprzewidywalnym opóźnieniem poza tym szacunkiem.

**Ścieżka B (community)**: nauka toolchaina MLIR-AIE/IRON od zera, skompilowanie i
odpalenie choćby jednego prostego modelu (nie mówiąc o MobileNetV3 z pełną rurą
pomiarową tego projektu) to realistycznie **kilka dni do ponad tygodnia** dla kogoś bez
wcześniejszego doświadczenia z MLIR/kompilatorami niskiego poziomu.

**Obie ścieżki przekraczają próg ~1 dnia** ustalony w zadaniu.

## Rekomendacja

**Pozostać przy wcześniejszej decyzji: NPU AMD/XDNA (Proksza) poza zakresem tej pracy.**
Udokumentuj to w rozdziale metodologicznym jako świadome ograniczenie — analogicznie do
istniejącego ograniczenia "RAPL na Windows" (`docs/measurement_methodology.md`):

> Wsparcie AMD Ryzen AI (XDNA) NPU zostało rozpoznane (research spike, wrzesień 2026) i
> świadomie wyłączone z zakresu pracy. Sterownik jądra `amdxdna` jest upstreamowany od
> Linux 6.14, ale ekosystem software'owy (oficjalny Vitis AI EP dla Linuksa — "early
> access", ograniczony do zarejestrowanych klientów AMD i Ubuntu 24.04; alternatywny
> otwarty stos IRON/MLIR-AIE — wymaga budowy własnego toolchaina kompilatorowego) nie
> pozwala na integrację proporcjonalną do budżetu czasowego tej pracy. W odróżnieniu od
> Intel NPU (dojrzały plugin OpenVINO, `device_name="NPU"`, gotowy do użycia), AMD XDNA
> wymagałby tygodni pracy integracyjnej, nie dni.

### Kiedy warto to zrewidować

Sytuacja **poprawia się aktywnie w 2026** (m.in. sierpniowy Ryzen AI Software 1.8.0) — to
nie jest "na zawsze zamknięte". Warto zrewidować, jeśli:

1. Early access do Ryzen AI Software na Linuksie stanie się publicznie dostępny (bez
   rejestracji) — sprawdź [stronę Ryzen AI Software](https://www.amd.com/en/products/software/ryzen-ai-software.html)
   od czasu do czasu.
2. Proksza ma już Ubuntu 24.04 LTS i jądro ≥6.14 (jeśli tak, warto samodzielnie sprawdzić
   `modinfo amdxdna` — to jedyny krok bez ryzyka, zajmuje minutę).
3. Harmonogram pracy magisterskiej ma zapas kilku dodatkowych dni pod koniec, gdzie
   eksploracyjny spike na Ścieżce B (community) mógłby być wartościowym dodatkiem, nawet
   bez pełnej integracji z resztą benchmarku.
