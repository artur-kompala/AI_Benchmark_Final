"""Komponent narracji - trzy zwiezle bloki pod KAZDYM wykresem w dashboardzie:
**Co pokazuje / Jak czytac / Co z tego wynika** (opcjonalnie: odniesienie do hipotezy).

Dashboard ma prowadzic czytelnika do wniosku, a nie wysypywac surowe wykresy - stad ten
blok jest obowiazkowy przy kazdej figurze i kazdej tabeli analitycznej."""

from __future__ import annotations

import streamlit as st

from dashboard.i18n.pl import (
    NARRATIVE_HOW,
    NARRATIVE_HYPOTHESIS,
    NARRATIVE_SO_WHAT,
    NARRATIVE_WHAT,
)


def interpretation(
    co_pokazuje: str,
    jak_czytac: str,
    co_wynika: str,
    hipoteza: str | None = None,
    *,
    expanded: bool = True,
) -> None:
    """Renderuje blok interpretacyjny pod wykresem. `hipoteza` (jesli podana) dostaje
    wlasny, wyrozniony wiersz - to jest pointa dla pracy."""
    with st.expander("Jak to czytac i co z tego wynika", expanded=expanded):
        st.markdown(
            f"**{NARRATIVE_WHAT}.** {co_pokazuje}\n\n"
            f"**{NARRATIVE_HOW}.** {jak_czytac}\n\n"
            f"**{NARRATIVE_SO_WHAT}.** {co_wynika}"
        )
        if hipoteza:
            st.markdown(f"> **{NARRATIVE_HYPOTHESIS}.** {hipoteza}")


def page_intro(text: str) -> None:
    """Krotki akapit-wprowadzenie na gorze strony (po tytule)."""
    st.markdown(text)


def data_note(text: str) -> None:
    """Drobna nota o danych/zastrzezeniach - mniej wyeksponowana niz interpretacja."""
    st.caption(text)
