-- 0003_experiment_checkpoints.sql
-- Faza opcjonalna: checkpointing macierzy eksperymentow, zeby dalo sie wznowic
-- przerwana serie (np. po awarii zasilania) bez powtarzania juz ukonczonych przebiegow.
-- Uzywane przez scripts/run_experiment_matrix.py (Faza 5).

create table experiment_checkpoints (
    id            uuid primary key default gen_random_uuid(),
    matrix_name   text not null,             -- nazwa serii eksperymentow (np. z pliku configu macierzy)
    config_hash   text not null,             -- sha256 pojedynczej kombinacji parametrow w macierzy
    status        text not null default 'pending',  -- 'pending' | 'running' | 'done' | 'failed'
    run_id        uuid references runs(id),  -- wypelniane po ukonczeniu przebiegu
    updated_at    timestamptz not null default now(),

    constraint uq_checkpoint unique (matrix_name, config_hash),
    constraint chk_checkpoint_status check (status in ('pending', 'running', 'done', 'failed'))
);

create index idx_checkpoints_matrix_status on experiment_checkpoints(matrix_name, status);
