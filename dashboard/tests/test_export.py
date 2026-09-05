"""Testy eksportu tabel - w szczegolnosci ze kolumny datetime ze strefa czasowa (UTC)
nie wywalaja eksportu do Excela (openpyxl nie obsluguje tz-aware datetimes)."""

from __future__ import annotations

import pandas as pd

from dashboard.components.export import dataframe_to_csv_bytes, dataframe_to_excel_bytes


def test_dataframe_to_excel_bytes_handles_timezone_aware_datetime():
    df = pd.DataFrame(
        {
            "started_at": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-02T11:00:00Z"], utc=True),
            "value": [1, 2],
        }
    )

    result = dataframe_to_excel_bytes(df)

    assert isinstance(result, bytes)
    assert len(result) > 0
    # oryginalny DataFrame nie powinien zostac zmutowany (nadal tz-aware)
    assert "UTC" in str(df["started_at"].dtype)


def test_dataframe_to_csv_bytes_handles_timezone_aware_datetime():
    df = pd.DataFrame({"started_at": pd.to_datetime(["2026-01-01T10:00:00Z"], utc=True)})
    result = dataframe_to_csv_bytes(df)
    assert isinstance(result, bytes)
    assert b"2026-01-01" in result
