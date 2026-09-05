"""Wspolna konfiguracja pytest - dodaje src/ do sys.path, zeby testy dzialaly
rowniez bez `pip install -e .` (np. szybkie uruchomienie lokalnie/w CI)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
