"""Warstwa dostepu do danych - CRUD dla tabel devices/runs/power_samples/run_summary."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from supabase import Client

from benchmark_runner.storage.models import (
    DeviceRecord,
    PowerSampleRecord,
    RunRecord,
    RunStatsSummaryRecord,
    RunSummaryRecord,
)

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Blad komunikacji z Supabase (siec, autoryzacja, ...). Orchestrator lapie ten
    wyjatek i przekierowuje zapis do LocalBuffer zamiast przerywac cala serie pomiarowa."""


def _to_supabase_payload(value: Any) -> Any:
    """Konwertuje wartosci niewspierane bezposrednio przez klienta (datetime -> ISO string)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_supabase_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_supabase_payload(v) for v in value]
    return value


def _format_supabase_error(exc: Exception) -> str:
    """Wyciaga PELNA tresc odpowiedzi bledu (np. 'violates check constraint chk_device_type'),
    nie tylko kod statusu HTTP - bez tego log pokazuje wylacznie '400 Bad Request', co
    uniemozliwia diagnoze (patrz storage/repository.py w kontekscie bledow zapisu do
    tabeli devices po dodaniu device_type='intel_gpu_openvino')."""
    detail: Any = None

    json_method = getattr(exc, "json", None)
    if callable(json_method):
        try:
            detail = json_method()
        except Exception:
            detail = None

    if detail is None:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                detail = response.json()
            except Exception:
                detail = getattr(response, "text", None)

    base = str(exc)
    # postgrest.exceptions.APIError juz wbudowuje pelna tresc bledu (message/code/hint/
    # details) w str(exc) - dopisujemy 'detail' osobno tylko gdy wnosi cos nowego (np.
    # httpx.HTTPStatusError, gdzie str(exc) to tylko status HTTP bez body odpowiedzi).
    if detail and str(detail) not in base:
        return f"{base} | tresc odpowiedzi Supabase: {detail}"
    return base


class RunRepository:
    def __init__(self, client: "Client") -> None:
        self._client = client

    def get_or_create_device(self, device: DeviceRecord) -> str:
        try:
            existing = (
                self._client.table("devices")
                .select("id")
                .eq("machine_name", device.machine_name)
                .eq("device_type", device.device_type)
                .eq("device_label", device.device_label)
                .limit(1)
                .execute()
            )
            if existing.data:
                return existing.data[0]["id"]

            payload = _to_supabase_payload(asdict(device))
            result = self._client.table("devices").insert(payload).execute()
            return result.data[0]["id"]
        except Exception as exc:
            raise RepositoryError(f"Nie udalo sie pobrac/utworzyc urzadzenia w Supabase: {_format_supabase_error(exc)}") from exc

    def insert_run(self, run: RunRecord) -> str:
        payload = _to_supabase_payload(asdict(run))
        try:
            result = self._client.table("runs").insert(payload).execute()
            return result.data[0]["id"]
        except Exception as exc:
            raise RepositoryError(f"Nie udalo sie zapisac przebiegu (run) w Supabase: {_format_supabase_error(exc)}") from exc

    def insert_power_samples_batch(self, samples: list[PowerSampleRecord]) -> None:
        if not samples:
            return
        payload = [_to_supabase_payload(asdict(s)) for s in samples]
        try:
            self._client.table("power_samples").insert(payload).execute()
        except Exception as exc:
            raise RepositoryError(f"Nie udalo sie zapisac probek mocy w Supabase: {_format_supabase_error(exc)}") from exc

    def insert_run_summary(self, summary: RunSummaryRecord) -> None:
        payload = _to_supabase_payload(asdict(summary))
        try:
            self._client.table("run_summary").insert(payload).execute()
        except Exception as exc:
            raise RepositoryError(f"Nie udalo sie zapisac podsumowania przebiegu w Supabase: {_format_supabase_error(exc)}") from exc

    def upsert_run_stats_summary(self, stats: RunStatsSummaryRecord) -> None:
        """upsert (nie insert) na repetition_group_id (PK) - orchestrator liczy to raz po
        zakonczeniu wszystkich powtorzen jednej kombinacji, ale idempotencja chroni przed
        duplikatami przy ewentualnym ponownym wywolaniu (np. wznowienie po awarii)."""
        payload = _to_supabase_payload(asdict(stats))
        try:
            self._client.table("run_stats_summary").upsert(payload).execute()
        except Exception as exc:
            raise RepositoryError(f"Nie udalo sie zapisac run_stats_summary w Supabase: {_format_supabase_error(exc)}") from exc
