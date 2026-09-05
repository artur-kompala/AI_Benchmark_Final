"""Formatowanie liczb po polsku - przecinek dziesietny, spacja (niełamliwa) jako
separator tysiecy. Uzywane wszedzie: etykiety na wykresach, tabele decyzyjne, KPI,
podpisy. Jeden modul, zeby format byl spojny w calym dashboardzie (patrz wymog
'liczby formatowane po polsku, rozsadna liczba miejsc po przecinku')."""

from __future__ import annotations

import math
from typing import Any

_NBSP = " "  # niełamliwa spacja - separator tysiecy, nie rozjezdza sie w komorce
_MISSING = "—"  # em dash - jawny brak wartosci (nie mylic z zerem)

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(math.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def _group_thousands(int_part: str) -> str:
    """'1234567' -> '1 234 567' (spacja niełamliwa)."""
    neg = int_part.startswith("-")
    digits = int_part[1:] if neg else int_part
    chunks = []
    while len(digits) > 3:
        chunks.insert(0, digits[-3:])
        digits = digits[:-3]
    chunks.insert(0, digits)
    return ("-" if neg else "") + _NBSP.join(chunks)


def fmt_pl(x: Any, decimals: int = 2) -> str:
    """Liczba po polsku: '1 234,56'. None/NaN -> '—'. Zaokragla do `decimals`."""
    if _is_missing(x):
        return _MISSING
    value = float(x)
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0  # zbija '-0,00'
    text = f"{rounded:.{decimals}f}"
    int_part, _, frac_part = text.partition(".")
    grouped = _group_thousands(int_part)
    return f"{grouped},{frac_part}" if decimals > 0 else grouped


def fmt_int_pl(x: Any) -> str:
    """Liczba calkowita po polsku: '1 234'. None/NaN -> '—'."""
    if _is_missing(x):
        return _MISSING
    return _group_thousands(f"{int(round(float(x)))}")


def fmt_ratio(x: Any, decimals: int = 1) -> str:
    """Krotnosc: 2.34 -> '2,3×'. Uzywane w porownaniach parami (ile razy oszczedniej)."""
    if _is_missing(x):
        return _MISSING
    return f"{fmt_pl(x, decimals)}×"


def fmt_pct_pl(x: Any, decimals: int = 1) -> str:
    """Procent: 52.83 -> '52,8%'. Wartosc juz w punktach procentowych (nie ulamek)."""
    if _is_missing(x):
        return _MISSING
    return f"{fmt_pl(x, decimals)}%"


def fmt_signed_pct_pl(x: Any, decimals: int = 0) -> str:
    """Procent ze znakiem: -18 -> '−18%', 5 -> '+5%'. Do delt vs punkt odniesienia."""
    if _is_missing(x):
        return _MISSING
    sign = "+" if float(x) >= 0 else "−"
    return f"{sign}{fmt_pl(abs(float(x)), decimals)}%"


def fmt_sci_pl(x: Any, sig: int = 1) -> str:
    """Notacja naukowa po polsku z mnoznikiem i potega w indeksie gornym:
    13355033 -> '1,3·10⁷'. Do FLOPS/W i FLOPS/probke (wartosci rzedu 10⁷–10⁹)."""
    if _is_missing(x):
        return _MISSING
    value = float(x)
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    # korekta zaokraglenia mantysy (np. 9,95 -> 10,0 -> 1,0·10^(n+1))
    if round(abs(mantissa), sig) >= 10:
        mantissa /= 10
        exponent += 1
    mantissa_txt = fmt_pl(mantissa, sig)
    return f"{mantissa_txt}·10{str(exponent).translate(_SUPERSCRIPT)}"


def fmt_range_pl(low: Any, high: Any, decimals: int = 2) -> str:
    """Przedzial: '0,12 – 3,40'. Do opisu rozrzutu pod wykresem."""
    return f"{fmt_pl(low, decimals)}{_NBSP}–{_NBSP}{fmt_pl(high, decimals)}"
