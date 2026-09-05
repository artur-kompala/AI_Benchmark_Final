"""Testy formatowania liczb po polsku (components/format.py)."""

from __future__ import annotations

import math

from dashboard.components.format import (
    fmt_int_pl,
    fmt_pct_pl,
    fmt_pl,
    fmt_ratio,
    fmt_sci_pl,
    fmt_signed_pct_pl,
)

_NBSP = " "


def test_fmt_pl_decimal_comma_and_thousands_space():
    assert fmt_pl(1234.5678, 2) == f"1{_NBSP}234,57"
    assert fmt_pl(0.1, 3) == "0,100"
    assert fmt_pl(-0.001, 2) == "0,00"  # zbija '-0,00'


def test_fmt_pl_missing_values():
    assert fmt_pl(None) == "—"
    assert fmt_pl(float("nan")) == "—"


def test_fmt_int_pl():
    assert fmt_int_pl(1263) == f"1{_NBSP}263"
    assert fmt_int_pl(None) == "—"


def test_fmt_ratio_and_pct():
    assert fmt_ratio(2.345) == "2,3×"
    assert fmt_pct_pl(52.83) == "52,8%"
    assert fmt_signed_pct_pl(-18) == "−18%"
    assert fmt_signed_pct_pl(5) == "+5%"


def test_fmt_sci_pl_superscript_exponent():
    assert fmt_sci_pl(13355033, 1) == "1,3·10⁷"
    assert fmt_sci_pl(9.99e6, 1) == "1,0·10⁷"  # zaokraglenie mantysy przenosi wykladnik
    assert fmt_sci_pl(0) == "0"
    assert fmt_sci_pl(None) == "—"
