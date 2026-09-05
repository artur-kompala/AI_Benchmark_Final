"""Faza 5 (opcjonalna): uruchamianie calej macierzy eksperymentow z checkpointingiem
w Supabase (tabela experiment_checkpoints, db/migrations/0003_experiment_checkpoints.sql),
zeby dalo sie wznowic przerwana serie bez powtarzania juz ukonczonych przebiegow.

W Fazie 2 do uruchamiania pojedynczego pliku configu (ktory sam w sobie moze definiowac
macierz przez listy models/devices/precisions/batch_sizes/phases) sluzy:
    python -m benchmark_runner run --config <plik.yaml>
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Uruchamianie wielu plikow configu z checkpointingiem zostanie zaimplementowane w Fazie 5"
    )


if __name__ == "__main__":
    main()
