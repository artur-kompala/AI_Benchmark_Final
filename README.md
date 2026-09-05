# AI Benchmark — CPU vs GPU vs NPU

Aplikacja badawcza na potrzeby pracy magisterskiej *"Analiza energooszczędności
i wydajności nowoczesnych układów obliczeniowych w zastosowaniach sztucznej
inteligencji"*. Porównuje CPU, GPU (NVIDIA/AMD) i NPU (Intel Core Ultra) pod
kątem czasu wykonania, mocy chwilowej i całkowitego zużycia energii podczas
treningu i wnioskowania modeli AI.

## Architektura

Projekt składa się z dwóch **całkowicie niezależnych** warstw, które nie dzielą
wspólnego procesu ani importowanego kodu:

- **[`benchmark_runner/`](benchmark_runner/README.md)** — headless CLI w Pythonie.
  Uruchamiany osobno na każdej maszynie testowej (PC z GPU NVIDIA, PC z GPU AMD,
  laptop z Intel NPU). Wykonuje eksperymenty zdefiniowane w plikach YAML, mierzy
  czas i energię, zapisuje wyniki do wspólnej, chmurowej bazy Supabase.
- **`dashboard/`** — aplikacja Streamlit tylko do odczytu. Czyta wyniki z
  Supabase i renderuje wykresy/rankingi po polsku. Nigdy nie uruchamia
  benchmarków (żeby nie zaburzać pomiarów mocy współbieżnym obciążeniem).

Jedynym punktem styku między warstwami jest schemat bazy danych w
[`db/migrations/`](db/migrations/).

## Struktura repozytorium

```
AI__Benchmark/
├── db/                  # migracje SQL Supabase (schemat wspoldzielonej bazy)
├── docs/                # dokumentacja: setup Supabase, GPU, NPU, metodologia pomiaru
├── benchmark_runner/    # warstwa 1: runner benchmarkow (CLI, headless)
└── dashboard/           # warstwa 2: dashboard analityczny (Streamlit)
```

## Szybki start

1. Załóż darmowy projekt Supabase i uruchom migracje SQL — patrz
   [`docs/setup_supabase.md`](docs/setup_supabase.md).
2. Skopiuj `.env.example` do `.env` w katalogu głównym i w `benchmark_runner/`,
   uzupełnij `SUPABASE_URL`/`SUPABASE_KEY`.
3. Skonfiguruj runner na maszynie testowej — patrz
   [`benchmark_runner/README.md`](benchmark_runner/README.md) oraz, zależnie od
   sprzętu, [`docs/setup_windows_gpu.md`](docs/setup_windows_gpu.md) lub
   [`docs/setup_npu_intel.md`](docs/setup_npu_intel.md).
4. Uruchom pierwszy eksperyment i sprawdź wyniki w tabeli `runs` w Supabase.
5. Uruchom dashboard, żeby zwizualizować zebrane dane — patrz [`dashboard/README.md`](dashboard/README.md).

## Dokumentacja

- [`docs/architecture.md`](docs/architecture.md) — architektura systemu i uzasadnienie decyzji projektowych.
- [`docs/setup_supabase.md`](docs/setup_supabase.md) — zakładanie projektu Supabase i uruchamianie migracji.
- [`docs/setup_windows_gpu.md`](docs/setup_windows_gpu.md) — konfiguracja PC z GPU NVIDIA/AMD na Windows.
- [`docs/setup_npu_intel.md`](docs/setup_npu_intel.md) — konfiguracja laptopa z Intel Core Ultra (NPU).
- [`docs/setup_intel_gpu.md`](docs/setup_intel_gpu.md) — konfiguracja maszyny z Intel GPU (zintegrowane/dyskretne).
- [`docs/measurement_methodology.md`](docs/measurement_methodology.md) — metodologia pomiaru energii (do rozdziału metodologicznego pracy).
- [`docs/setup_smart_plug.md`](docs/setup_smart_plug.md) — konfiguracja watomierza fizycznego Tuya (np. ATORCH S1).

## Status implementacji

- ✅ Faza 1: struktura projektu, schemat SQL Supabase.
- ✅ Faza 2: runner dla klasyfikacji obrazów (MobileNetV3/CIFAR-10) na CPU/CUDA/ROCm, zapis do Supabase.
- ✅ Faza 3: ścieżka NPU (OpenVINO / ONNX Runtime DirectML) — tylko inferencja, pomiar mocy jako fallback whole-system.
- ✅ Faza 4: watomierz fizyczny Tuya (np. ATORCH S1) — automatyczny odczyt lokalny (`tinytuya`), ciągłe próbkowanie mocy + odczyt licznika energii urządzenia; tryb manualny jako fallback.
- ✅ Faza 5: zadanie NLP (DistilBERT/IMDB fine-tuning+inferencja), LLM inference (ONNX Runtime + llama-cpp-python, fp32/fp16/int8), dashboard Streamlit (5 widoków, PL).
- ✅ Faza 6: drugi model klasyfikacji obrazów (ResNet-50/CIFAR-10, zaadaptowany pod 32x32), powtórzenia pomiarów + statystyka (mean/std/coefficient of variation, tabela `run_stats_summary`), kwantyzacja INT8 (dynamiczna, PyTorch) rozszerzona na `image_classification`/`nlp_sentiment`, rozróżnienie GPU dyskretne/zintegrowane, przebudowany dashboard (6 widoków).
