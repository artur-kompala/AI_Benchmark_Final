# Konfiguracja watomierza fizycznego (Tuya, np. ATORCH S1-B/W/T/H)

Runner obsługuje automatyczny odczyt mocy i energii z inteligentnych gniazdek Tuya
przez lokalny protokół (biblioteka [`tinytuya`](https://github.com/jasonacox/tinytuya))
— po jednorazowej konfiguracji działa **całkowicie lokalnie**, bez chmury Tuya podczas
samego pomiaru. Jeśli nie skonfigurujesz tych danych, runner automatycznie wraca do
trybu manualnego (ręczne wpisywanie odczytu) — nic się nie wywala.

## 1. Wymagania wstępne

- Gniazdko musi być sparowane z aplikacją **Smart Life** lub **Tuya Smart** i podłączone
  do tej samej sieci WiFi co komputer uruchamiający runner.
- Zainstaluj `tinytuya` (jest już w `requirements-common.txt`, więc jeśli masz
  zainstalowane zależności runnera, masz to automatycznie):

  ```bash
  pip install tinytuya
  ```

## 2. Pozyskanie `device_id`, `local_key` i adresu IP

Tuya nie udostępnia `local_key` (klucza do lokalnego sterowania) bez przejścia przez
ich chmurę **raz**, żeby go wydobyć — to jednorazowa operacja, później działa lokalnie.

1. Załóż darmowe konto na [Tuya IoT Platform](https://iot.tuya.com) (Cloud Development).
2. Utwórz **Cloud Project** (Development → Create Cloud Project), typ dowolny (np.
   "Smart Home"), region zgodny z tym, którego używa Twoja aplikacja Smart Life.
3. W zakładce **Devices** projektu połącz swoje konto Smart Life/Tuya Smart (Link
   Tuya App Account) — Twoje sparowane urządzenia (w tym gniazdko) powinny się
   pojawić na liście.
4. Uruchom wbudowany kreator `tinytuya`, który przeprowadzi Cię przez resztę i zapisze
   dane lokalnie:

   ```bash
   python -m tinytuya wizard
   ```

   Wpisz `API Key`, `API Secret` i `region` ze swojego Cloud Project (Overview →
   Authorization Key), gdy zapyta. Kreator zapisze `devices.json` ze wszystkimi
   Twoimi urządzeniami, w tym `id` (device_id) i `key` (local_key).
5. Znajdź w `devices.json` wpis odpowiadający Twojemu gniazdku (po nazwie z aplikacji
   Smart Life) i zanotuj `id` → `TUYA_DEVICE_ID`, `key` → `TUYA_LOCAL_KEY`.

   **Nie myl `key` (local_key urządzenia) z `apiKey`/`apiSecret` z `tinytuya.json`**
   — to są dane logowania do Twojego konta Tuya Cloud (używane tylko przez kreatora),
   nie klucz do lokalnej komunikacji z konkretnym urządzeniem. Łatwo je pomylić, bo
   oba wygladaja jak losowe ciagi znakow.

   **Pole `ip` w `devices.json`/`tuya-raw.json` NIE jest wiarygodne** — to adres
   zgłoszony do chmury Tuya, który przez NAT bywa Twoim publicznym adresem WAN, a nie
   lokalnym adresem LAN potrzebnym do protokołu lokalnego (`192.168.x.x`/`10.x.x.x`).
   Nie wpisuj tej wartości jako `TUYA_IP_ADDRESS`.

6. Zamiast tego znajdź prawdziwy lokalny adres IP. Najpierw spróbuj pasywnego
   nasłuchu (szybsze, ~18 sekund):

   ```bash
   python -m tinytuya scan
   ```

   **Jeśli to nie znajdzie urządzenia** (0 devices found — niektóre modele, w tym
   potwierdzone ATORCH S1BW, nie wysyłają własnych rozgłoszeń UDP), użyj aktywnego
   skanowania całej podsieci zamiast pasywnego nasłuchu:

   ```bash
   python -m tinytuya scan -force -yes
   ```

   To aktywnie odpytuje kolejno każdy adres w Twojej podsieci (auto-wykrywa
   maskę/zakres) - trwa dłużej (do kilkudziesięciu sekund), ale znajduje urządzenia,
   które nie rozgłaszają się same. Wypisze `Address = ...` dla znalezionego
   urządzenia razem z `Version` (protokołu - moze się różnić od domyślnego `3.3`,
   np. `3.5`) - zanotuj oba.

   **Jeśli nawet `-force` nic nie znajdzie** — sprawdź w kolejności:
   - Czy komputer i gniazdko są faktycznie na tej samej sieci WiFi/VLAN? Wiele
     routerów mesh (i niektóre ISP) domyślnie umieszcza urządzenia IoT paczkowane
     przez appki typu Smart Life na osobnej sieci "IoT"/"Gość" z izolacją klientów
     (AP isolation) - to najczęstsza przyczyna.
   - Czy port TCP 6668 (lokalne sterowanie) i UDP 6666/6667/7000 (discovery) nie są
     blokowane przez firewall (Windows Defender Firewall → dozwolone aplikacje,
     sprawdź Python/venv)?
   - Sprawdź panel administracyjny routera (lista klientów DHCP) i poszukaj po adresie
     MAC urządzenia (widoczny w `tuya-raw.json` jako `mac`) - to najpewniejszy sposób,
     niezależny od tinytuya.

Jeśli adres IP się zmieni w przyszłości (DHCP), uruchom ponownie skan, żeby znaleźć
nowy, albo przypisz gniazdku statyczny IP w ustawieniach routera (zalecane, żeby
konfiguracja w `.env` nie wymagała aktualizacji).

## 3. Konfiguracja `.env`

W `benchmark_runner/.env` (skopiowanym z `.env.example`) uzupełnij:

```bash
TUYA_DEVICE_ID=twoj-device-id
TUYA_LOCAL_KEY=twoj-local-key
TUYA_IP_ADDRESS=192.168.x.x
TUYA_VERSION=3.3
```

`TUYA_VERSION` to wersja protokołu Tuya — większość nowszych urządzeń (w tym
prawdopodobnie ATORCH S1) używa `3.3` lub `3.4`. Jeśli połączenie się nie udaje,
spróbuj `3.4` (kreator `tinytuya wizard` zwykle też ją wskazuje).

## 4. Weryfikacja połączenia

```bash
python scripts/check_device_availability.py
```

Sekcja "Watomierz fizyczny" powinna pokazać `tuya (IP): DOSTEPNE`. Jeśli pokazuje
"BLAD POLACZENIA", sprawdź IP/local_key/wersję protokołu.

## 5. Weryfikacja DPS (WAŻNE — zrób to przed pierwszym pomiarem referencyjnym)

**DPS (data points) różnią się między urządzeniami**, nawet w tej samej kategorii
produktowej Tuya. Domyślne wartości w kodzie (`DPS_POWER="19"`, `DPS_ENERGY="17"`,
skale `0.1`/`0.01`) są typowe dla wtyczek z pomiarem energii, ale **nie są
gwarantowane** dla ATORCH S1 — musisz je zweryfikować:

```bash
python scripts/tuya_dps_scan.py
```

Skrypt co sekundę wypisuje pełny surowy słownik DPS z urządzenia. Włącz/wyłącz
odbiornik podłączony do gniazdka albo zmień jego obciążenie i obserwuj:

- które pole **zmienia się w takt zmiany poboru mocy** → to jest Twoje `TUYA_DPS_POWER`,
- które pole **rośnie monotonicznie w czasie** (nigdy nie maleje, chyba że
  zresetujesz licznik w aplikacji) → to jest Twoje `TUYA_DPS_ENERGY`.

Sprawdź też **jednostkę** — porównaj wartość z tego co pokazuje aplikacja Smart
Life w tym samym momencie (np. jeśli aplikacja pokazuje "23.5 W" a surowy DPS to
`235`, skala to `0.1`; jeśli DPS to `2350`, skala to `0.01`, itd.).

Jeśli DPS lub skala różnią się od domyślnych, nadpisz je w `.env`:

```bash
TUYA_DPS_POWER=9
TUYA_DPS_ENERGY=6
TUYA_POWER_SCALE=1.0
TUYA_ENERGY_SCALE=0.001
```

**Złe DPS/skala dają cicho błędne dane** (nie błąd, tylko złe liczby) — ten krok
weryfikacyjny jest ważny dla wiarygodności pomiarów w pracy, nie pomijaj go.

### Znane urządzenie: ATORCH Smart Socket S1BW (zweryfikowane na żywo)

Schemat DPS zgłoszony przez Tuya Cloud dla modelu `S1BW` (z `devices.json` po
uruchomieniu kreatora) i **potwierdzony na żywo** przez `tuya_dps_scan.py`
(napięcie 236V zgadza się z siecią EU przy skali 0.01 — zob. niżej):

| DPS | Pole | Jednostka | Skala | Uwaga |
|---|---|---|---|---|
| `1` | `switch_1` | bool | - | stan przekaźnika |
| `9` | `countdown_1` | s | - | timer, nieuzywane tutaj |
| `18` | `cur_current` | A | `0.001` (scale 3) | prąd chwilowy |
| `19` | `cur_power` | W | **`0.01`** (scale 2) | moc chwilowa - `TUYA_DPS_POWER=19`, `TUYA_POWER_SCALE=0.01` |
| `20` | `cur_voltage` | V | `0.01` (scale 2) | napięcie chwilowe - zweryfikowane: raw `23612` → 236.12 V |

Realny `status()` zwraca też dodatkowe pola `101`-`141` (np. `'107': 'english'`,
`'112': 'controlled'`, `'117': 'measurement'`, `'136': 'single_rate'`) - to
parametry konfiguracyjne/kalibracyjne urządzenia (język, tryb pracy, taryfa), nie
pomiary - żadne z nich nie zmieniało się w kilkusekundowym oknie obserwacji mimo
że to jedyne "podejrzane" kandydaty na licznik energii.

**Ten model NIE zgłasza żadnego DPS ze skumulowaną energią** (brak odpowiednika
`add_ele`) — potwierdzone zarówno w schemacie Tuya Cloud, jak i w realnym
`status()` urządzenia. Runner **automatycznie** oblicza
`run_summary.energy_joules_smart_plug` przez całkowanie próbek mocy chwilowej
(`source='smart_plug'`) zamiast z licznika urządzenia - nie trzeba nic dodatkowo
konfigurować, to fallback wbudowany w `core/orchestrator.py`. Jedyna różnica:
wartość jest wtedy wyliczona z próbkowania (tak jak `energy_joules_software`), a nie
odczytana z wewnętrznego akumulatora sprzętowego — warto to odnotować w rozdziale
metodologicznym pracy jako różnicę względem modeli z DPS energii.

**Adres IP w `devices.json`/`tuya-raw.json` dla tego urządzenia to adres publiczny
(WAN), nie lokalny** — lokalny adres znajdziesz tylko przez aktywne skanowanie
podsieci:

```bash
python -m tinytuya scan -force -yes
```

(zwykły `python -m tinytuya scan` bez `-force` nasłuchuje pasywnie na rozgłoszenia
UDP z urządzenia - ten konkretny model ich nie wysyła, więc pasywny scan zawsze
znajdzie 0 urządzeń dla tego modelu; wariant `-force` aktywnie odpytuje każdy host
w podsieci, więc działa niezależnie od tego).

## 6. Włączenie w configu eksperymentu

W pliku YAML configu (np. `configs/experiments/CPU/image_classification_mobilenet_cifar10.yaml`):

```yaml
power_meter:
  smart_plug:
    backend: "tuya"
    enabled: true
```

`device_id`/`local_key`/`ip` **celowo nie są tu** — to sekrety, żyją tylko w
`.env` (gitignored), żeby nie trafiły do wersjonowanych plików YAML w
`configs/experiments/`.

## 7. Co się dzieje podczas przebiegu

- **Moc chwilowa**: gniazdko jest odpytywane w tym samym interwale co inne źródła
  (`power_meter.sampling_interval_ms`, domyślnie 100ms) i trafia do `power_samples`
  z `source='smart_plug'` — widoczne w dashboardzie obok RAPL/NVML/itd.
- **Energia skumulowana**: na starcie i końcu przebiegu runner odczytuje **wewnętrzny
  licznik energii urządzenia** (nie integruje próbek mocy) — to trafia do
  `run_summary.energy_joules_smart_plug` jako wartość autorytatywna, niezależna od
  jakichkolwiek błędów próbkowania programowego. Stąd `power_discrepancy_pct` — patrz
  [`measurement_methodology.md`](measurement_methodology.md).
- Jeśli gniazdko przestanie odpowiadać w trakcie serii (np. utrata WiFi), pojedyncze
  próbki/odczyty zwracają `None` i są pomijane — benchmark **kontynuuje**, tylko bez
  danych z tego źródła dla dotkniętych przebiegów (zgodnie z ogólną zasadą "brak
  źródła pomiaru nie przerywa benchmarku").

## Rozwiązywanie problemów

- **"BLAD POLACZENIA"** — najczęściej zły adres IP (DHCP mógł go zmienić od czasu
  `tinytuya wizard`) albo zła wersja protokołu. Uruchom `python -m tinytuya scan`,
  żeby znaleźć aktualny IP.
- **Wartości mocy/energii wyglądają nierealistycznie** (np. 10000x za duże/małe) —
  zła skala (`TUYA_POWER_SCALE`/`TUYA_ENERGY_SCALE`) — wróć do kroku 5.
- **Energia skumulowana "resetuje się"/maleje** — runner sam to wykrywa i odrzuca
  pomiar dla tego przebiegu (log ostrzegawczy), zamiast zapisać ujemną wartość.
