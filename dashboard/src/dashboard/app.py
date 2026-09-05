"""Dashboard analityczny do pracy magisterskiej - router stron (Streamlit `st.navigation`).

Uruchom:  streamlit run src/dashboard/app.py

Zrodlo danych: WYLACZNIE Supabase na zywo (data_access/queries.py, cache ttl=60).
Dashboard nigdy nie zapisuje do bazy ani nie uruchamia benchmarkow - to zadanie
benchmark_runner (osobny, niezalezny proces; patrz ../../docs/architecture.md).
Bez trybu offline, bez importu z plikow, bez przelacznika zrodla - dane wciaz przyrastaja,
wiec dashboard ma zawsze pokazywac aktualny stan z Supabase.
"""

from __future__ import annotations

import streamlit as st

from dashboard.i18n.pl import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="⚡")

_PAGES = [
    ("views/0_start.py", "0 · Start / Jak czytac", "🧭"),
    ("views/1_efektywnosc.py", "1 · Efektywnosc energetyczna", "⚡"),
    ("views/2_wydajnosc.py", "2 · Wydajnosc", "🚀"),
    ("views/3_pareto.py", "3 · Kompromis wydajnosc / energia", "⚖️"),
    ("views/4_batch_size.py", "4 · Wplyw rozmiaru partii", "📦"),
    ("views/5_precyzja.py", "5 · Wplyw precyzji (fp32/fp16/int8)", "🎯"),
    ("views/6_moc_temperatura.py", "6 · Moc chwilowa i temperatura", "🌡️"),
    ("views/7_para_maszyna.py", "7 · CPU vs akcelerator (ta sama maszyna)", "🤝"),
    ("views/8_pomiar.py", "8 · Co mierzymy / powtarzalnosc", "🔬"),
    ("views/9_hipoteza.py", "9 · Weryfikacja hipotezy / wnioski", "✅"),
    ("views/10_eksport.py", "10 · Eksport danych", "💾"),
]

nav = st.navigation(
    [st.Page(path, title=title, icon=icon, default=(i == 0)) for i, (path, title, icon) in enumerate(_PAGES)]
)
nav.run()
