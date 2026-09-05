# Konfiguracja realnego pomiaru mocy CPU na AMD Zen (zenpower3)

> **Status**: opcjonalne. Runner działa bez tego kroku — bez zainstalowanego modułu
> `zenpower3` `ZenpowerPowerSampler.is_available()` czysto zwraca `False`, a
> `power_meter.cpu_sources` (domyślnie `["zenpower", "rapl", "codecarbon"]`) automatycznie
> przechodzi na `rapl` (jeśli dostępny) albo `codecarbon` (estymacja z TDP × obciążenie,
> zawsze dostępna, ale to **nie jest pomiar fizyczny**). Ten dokument opisuje, jak
> zainstalować `zenpower3`, żeby uzyskać **realny** pomiar mocy CPU (telemetria SVI2 z VRM)
> zamiast estymacji.

## Dlaczego to jest potrzebne

RAPL (`Running Average Power Limit`, natywnie interfejs Intela) jest na AMD niekompletny/
niedostępny w jądrze w zależności od generacji CPU i konfiguracji kernela:

- Na niektórych maszynach AMD węzeł RAPL w ogóle nie istnieje w `/sys/class/powercap`.
- Na innych istnieje, ale odczyt `energy_uj` jest zablokowany uprawnieniami
  (`PermissionError`, plik `-r--------` należący do roota) — potwierdzone empirycznie na
  kilku maszynach testowych tego projektu (zarówno Ryzen 7000-series, jak i 5000-series).

Bez `zenpower3`, jedynym pozostałym źródłem dla CPU jest wtedy `codecarbon` w trybie
fallback (estymacja z TDP × obciążenie procesora) — przydatne jako ostatnia deska
ratunku, ale nie jest to pomiar fizyczny i nie nadaje się jako podstawa mocnych wniosków
w pracy o rzeczywistym poborze mocy.

`zenpower3` (fork [`zenpower`](https://github.com/Ta180m/zenpower3) z obsługą Zen 3+) to
moduł jądra, który czyta rzeczywistą telemetrię SVI2 (napięcie/prąd/moc) bezpośrednio z
VRM przez `amd_smn_read` i wystawia ją przez `hwmon` w sysfs — to jest **realny pomiar
sprzętowy**, nie estymacja.

## Wymagania

- CPU AMD Zen (Zen 1 przez oryginalny `zenpower`, Zen 3+ przez fork `zenpower3`).
- Linux (moduł jądra — brak odpowiednika na Windows).
- Uprawnienia administratora do instalacji modułu przez DKMS.

## Instalacja (krok manualny — kod tego repo tego nie robi za Ciebie)

```bash
sudo apt update
sudo apt install -y dkms git build-essential linux-headers-$(uname -r)

git clone https://github.com/Ta180m/zenpower3.git
cd zenpower3
sudo make dkms-install
```

### Zablokuj `k10temp` (współdzieli to samo urządzenie PCI co zenpower)

`k10temp` (wbudowany w jądro sterownik temperatury AMD) i `zenpower3` konkurują o ten sam
adres PCI — jeśli oba są załadowane, `zenpower3` może się nie załadować poprawnie lub
dawać błędne odczyty. Zablokuj `k10temp`:

```bash
echo "blacklist k10temp" | sudo tee /etc/modprobe.d/zenpower.conf
sudo modprobe -r k10temp   # jesli juz zaladowany - usun z biezacej sesji
```

### Restart

```bash
sudo reboot
```

## Weryfikacja

```bash
sensors | grep -A5 zenpower
```

albo bezpośrednio przez Python w aktywnym venv projektu:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from benchmark_runner.power.zenpower_meter import ZenpowerPowerSampler
s = ZenpowerPowerSampler()
print('dostepne:', s.is_available())
print('probka:', s.sample())
"
```

Albo po prostu:

```bash
python scripts/check_device_availability.py
```

Powinno pokazać `zenpower (CPU, AMD Zen): DOSTEPNE`.

## Uwaga o dokładnej nazwie atrybutu hwmon

`ZenpowerPowerSampler` szuka wpisu `hwmon` o nazwie `zenpower`, a następnie próbuje po
kolei atrybutów `power1_input`, `power1_average` — dokładna nazwa atrybutu (i to, który z
nich w ogóle jest udostępniony) różni się między wersjami modułu i modelami CPU. Jeśli
`is_available()` zwraca `False` mimo załadowanego modułu, sprawdź ręcznie:

```bash
ls /sys/class/hwmon/hwmon*/name | xargs -I{} sh -c 'echo {}: $(cat {})' | grep zenpower
# znajdz sciezke, potem:
ls /sys/class/hwmon/hwmonX/   # X = numer znaleziony wyzej
```

Jeśli katalog istnieje, ale żaden z `power1_input`/`power1_average` nie jest obecny,
dopisz właściwą nazwę do `_POWER_ATTR_CANDIDATES` w
[`power/zenpower_meter.py`](../benchmark_runner/src/benchmark_runner/power/zenpower_meter.py).

## Rozwiązywanie problemów

- **Moduł się nie kompiluje** — sprawdź, czy `linux-headers-$(uname -r)` faktycznie
  odpowiada uruchomionemu jądru (`uname -r`), nie tylko najnowszemu zainstalowanemu.
- **`sensors` nie pokazuje `zenpower` mimo udanej instalacji** — uruchom
  `sudo sensors-detect` (część pakietu `lm-sensors`) i potwierdź wszystkie pytania
  domyślną odpowiedzią.
- **`is_available()` nadal `False` po restarcie** — sprawdź `lsmod | grep zenpower` (czy
  moduł jest w ogóle załadowany) i `dmesg | grep -i zenpower` (błędy przy ładowaniu, np.
  konflikt z nadal załadowanym `k10temp`).
- **Nie chcę tego instalować** — nic się nie psuje. Runner automatycznie użyje
  `rapl`/`codecarbon`, tak jak dotychczas.
