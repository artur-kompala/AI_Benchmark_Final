# Konfiguracja Supabase

Wyniki wszystkich maszyn testowych trafiają do jednej, współdzielonej,
chmurowej bazy Supabase (Postgres). Poniżej instrukcja założenia darmowego
projektu i uruchomienia schematu.

## 1. Załóż konto i projekt

1. Wejdź na [supabase.com](https://supabase.com) i załóż darmowe konto
   (wystarczy GitHub/Google/e-mail).
2. Kliknij **New Project**.
3. Wybierz organizację, nadaj projektowi nazwę (np. `ai-benchmark-thesis`),
   ustaw hasło do bazy danych (**zapisz je** — będzie potrzebne, jeśli
   kiedykolwiek zechcesz połączyć się bezpośrednio przez `psql`) i wybierz
   region (najbliższy geograficznie, żeby zminimalizować opóźnienia zapisu).
4. Poczekaj aż projekt się utworzy (zwykle 1–2 minuty).

Darmowy tier Supabase (Free Plan) w zupełności wystarcza do potrzeb tego
projektu (baza jest odczytywana/zapisywana głównie przez runner i dashboard,
ruch jest niewielki).

## 2. Uzyskaj URL i klucz API

1. W panelu projektu wejdź w **Project Settings → API**.
2. Skopiuj:
   - **Project URL** → to jest `SUPABASE_URL`.
   - **`service_role` secret key** → to jest `SUPABASE_KEY` dla runnera
     (klucz z pełnymi prawami zapisu, omija Row Level Security).
   - Opcjonalnie: **`anon` public key** — jeśli w przyszłości dashboard
     będzie miał ograniczone prawa tylko-do-odczytu przez RLS, użyj tego
     klucza w `dashboard/.env` zamiast `service_role`.

3. Skopiuj `.env.example` (w katalogu głównym repo i w `benchmark_runner/`)
   do `.env` i uzupełnij oba pola:

   ```bash
   cp .env.example .env
   ```

   **Nigdy nie commituj `.env` do repozytorium** — jest w `.gitignore`.
   `service_role` key daje pełny dostęp do bazy, więc traktuj go jak hasło.

## 3. Uruchom migracje SQL

1. W panelu Supabase wejdź w **SQL Editor**.
2. Otwórz `db/migrations/0001_init_schema.sql` z tego repozytorium, skopiuj
   całą zawartość, wklej do edytora SQL i kliknij **Run**.
3. Powtórz dla **wszystkich kolejnych plików w katalogu `db/migrations/`, w
   kolejności numerów, aż do najwyższego numeru obecnego w repo** —
   `0002_indexes.sql`, `0003_experiment_checkpoints.sql`,
   `0004_repetitions_stats_quantization.sql`, `0005_accuracy.sql`,
   `0006_intel_gpu.sql`, itd. **Nie pomijaj żadnej** — nawet jeśli akurat nie
   korzystasz z funkcji, którą dodaje (np. `0003` pod macierz eksperymentów),
   pominięcie migracji, która rozszerza `check`-constrainty (np. `0006` dodaje
   `device_type='intel_gpu_openvino'` do `devices`), spowoduje że runner
   dostanie `400 Bad Request` przy zapisie dla urządzeń/wartości dodanych w
   tej migracji, mimo że kod runnera już je obsługuje.
4. Migracje uruchamiaj **w kolejności numerów** — każda kolejna zakłada, że
   poprzednie już się wykonały (np. `0003` odwołuje się do tabeli `runs`
   utworzonej w `0001`).
5. Ten plik dokumentacji bywa aktualizowany wolniej niż `db/migrations/` —
   przed uznaniem konfiguracji za skończoną zawsze sprawdź w repo, czy nie
   pojawiły się nowsze pliki `NNNN_*.sql`, niewymienione tutaj z nazwy.

## 4. Zweryfikuj schemat

1. Wejdź w **Table Editor** — powinieneś zobaczyć tabele: `devices`, `runs`,
   `power_samples`, `run_summary` (oraz `experiment_checkpoints`, jeśli
   uruchomiono `0003`).
2. Sprawdź w **Database → Indexes**, że indeksy z `0001`/`0002` istnieją
   (np. `idx_runs_task_device`).

## 5. (Opcjonalnie) Dane przykładowe do testów dashboardu

`db/seed/seed_devices_example.sql` wstawia dwa przykładowe wiersze do
`devices`, przydatne przy pierwszym uruchomieniu dashboardu zanim runner
wygeneruje realne dane. **Nie uruchamiaj tego na bazie z prawdziwymi
wynikami pomiarowymi** — służy wyłącznie do developmentu UI.

## Rozwiązywanie problemów

- **`relation "runs" does not exist` przy uruchamianiu `0003`** — nie
  uruchomiono najpierw `0001_init_schema.sql`.
- **Runner zgłasza błąd autoryzacji przy zapisie** — sprawdź, czy w `.env`
  jest `service_role` key, a nie `anon` (klucz `anon` bez odpowiednich
  polityk RLS nie ma prawa zapisu).
- **`400 Bad Request` przy zapisie do `devices` (np.
  `violates check constraint "chk_device_type"`)** — brakuje migracji, która
  rozszerzyła `check`-constraint o wartość `device_type`/`source`, której
  próbuje użyć runner (np. `intel_gpu_openvino` dodane w
  `0006_intel_gpu.sql`). Uruchom w SQL Editor wszystkie migracje z
  `db/migrations/`, których jeszcze nie uruchomiono (patrz sekcja 3 wyżej),
  **w kolejności numerów**. Log runnera od teraz pokazuje pełną treść
  odpowiedzi Supabase (nie tylko kod statusu) — sprawdź komunikat WARNING w
  logu, żeby zobaczyć dokładnie, którego constraintu dotyczy błąd. Wyniki
  zapisane w międzyczasie lokalnie (`.local_buffer/pending_runs.jsonl`) nie
  giną — po naprawieniu schematu uruchom `python scripts/flush_local_buffer.py`,
  żeby wysłać je do Supabase.
- **Wolne zapisy `power_samples`** (setki próbek na przebieg) — runner
  zapisuje próbki wsadowo (`insert_power_samples_batch`), a nie pojedynczo;
  jeśli mimo to zapis jest wolny, sprawdź opóźnienie sieciowe do wybranego
  regionu Supabase.
