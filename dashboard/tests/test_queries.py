"""Testy warstwy zapytan (offline, atrapa klienta Supabase - patrz conftest.py)."""

from __future__ import annotations

from dashboard.data_access.queries import _fetch_all, _flatten_run_row, get_runs_df
from tests.conftest import FakeClient, FakeQuery


def test_flatten_run_row_merges_device_and_summary_dict():
    row = {
        "id": "run-1",
        "task": "image_classification",
        "devices": {"id": "dev-1", "device_label": "CPU", "device_type": "cpu"},
        "run_summary": {"run_id": "run-1", "energy_joules_software": 12.5},
    }
    flat = _flatten_run_row(dict(row))
    assert flat["device_device_label"] == "CPU"
    assert flat["device_row_id"] == "dev-1"
    assert flat["summary_energy_joules_software"] == 12.5
    assert "devices" not in flat and "run_summary" not in flat
    assert "summary_run_id" not in flat


def test_flatten_run_row_handles_summary_as_list_defensively():
    row = {"id": "run-1", "devices": {"device_label": "CPU"},
           "run_summary": [{"run_id": "run-1", "energy_joules_software": 12.5}]}
    assert _flatten_run_row(dict(row))["summary_energy_joules_software"] == 12.5


def test_flatten_run_row_handles_missing_device_and_summary():
    row = {"id": "run-2", "task": "image_classification"}
    assert _flatten_run_row(dict(row)) == row


def test_get_runs_df_returns_empty_dataframe_when_no_rows():
    assert get_runs_df(FakeClient({"runs": []})).empty


def test_get_runs_df_flattens_parses_timestamps_and_enriches(runs_rows):
    df = get_runs_df(FakeClient({"runs": runs_rows}))
    assert len(df) == len(runs_rows)
    assert str(df["started_at"].dtype).startswith("datetime64")
    # kolumny pochodne z enrich
    for col in ["arch_class", "device_type_label", "energy_per_sample_j", "samples_per_joule",
                "is_whole_system", "scope_label", "device_short"]:
        assert col in df.columns


def test_fetch_all_paginates_beyond_1000_rows():
    """PostgREST oddaje maks. 1000 wierszy - _fetch_all musi skleic wszystkie strony."""
    rows = [{"id": f"r{i}"} for i in range(2500)]
    got = _fetch_all(lambda: FakeQuery(rows), page_size=1000)
    assert len(got) == 2500
    assert got[0]["id"] == "r0" and got[-1]["id"] == "r2499"


def test_fetch_all_single_page_when_small():
    rows = [{"id": f"r{i}"} for i in range(10)]
    assert len(_fetch_all(lambda: FakeQuery(rows), page_size=1000)) == 10
