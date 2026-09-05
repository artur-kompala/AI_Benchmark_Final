-- 0004_repetitions_stats_quantization.sql
-- Faza 6 (poprawki): powtorzenia pomiarow + statystyka (Poprawka 2), flaga sukcesu
-- kwantyzacji INT8 (Poprawka 3), kategoria GPU dyskretne/zintegrowane (dashboard).

-- ============================================================
-- runs: grupowanie powtorzen tej samej kombinacji parametrow
-- ============================================================
alter table runs add column repetition_group_id uuid not null default gen_random_uuid();
alter table runs add column repetition_index integer not null default 1;

comment on column runs.repetition_group_id is 'Wspolny dla wszystkich powtorzen (repetitions) tej samej kombinacji model/device/precision/batch_size/phase - patrz run_stats_summary';
comment on column runs.repetition_index is 'Numer powtorzenia w ramach repetition_group_id, liczony od 1';

create index idx_runs_repetition_group on runs(repetition_group_id);

-- ============================================================
-- run_summary: status kwantyzacji INT8 (Poprawka 3)
-- ============================================================
alter table run_summary add column quantization_status text;
alter table run_summary add constraint chk_run_summary_quantization_status check (
    quantization_status in ('success', 'partial', 'failed') or quantization_status is null
);

comment on column run_summary.quantization_status is 'Tylko dla precision=int8: success = wszystkie kwalifikujace sie warstwy skwantyzowane dynamicznie, partial = tylko czesc (np. Linear tak, Conv2d nie wspierane przez torch.quantization.quantize_dynamic), failed = kwantyzacja sie nie powiodla (uruchomiono fp32). NULL dla innych precyzji.';

-- ============================================================
-- devices: kategoria GPU dyskretne vs zintegrowane (iGPU)
-- ============================================================
alter table devices add column gpu_category text;
alter table devices add constraint chk_devices_gpu_category check (
    gpu_category in ('discrete', 'integrated') or gpu_category is null
);

comment on column devices.gpu_category is 'Tylko dla device_type=cuda/rocm, heurystyka po nazwie karty (devices/gpu_classification.py) - NULL dla cpu/npu lub gdy nie dotyczy';

-- ============================================================
-- run_stats_summary
-- Agregacja (mean/std/coefficient of variation) po WSZYSTKICH powtorzeniach jednej
-- kombinacji parametrow - liczona w Pythonie po stronie runnera
-- (orchestrator.ExperimentRunner._finalize_repetition_group), nie widok SQL, zeby
-- logika agregacji zyla w jednym miejscu z reszta pipeline'u metryk.
-- ============================================================
create table run_stats_summary (
    repetition_group_id            uuid primary key,

    task                             text not null,
    model_name                        text not null,
    device_id                          uuid not null references devices(id) on delete cascade,
    precision                           text not null,
    batch_size                           integer,
    phase                                 text not null,

    n_repetitions                         integer not null,

    mean_duration_s                        numeric,
    std_duration_s                          numeric,
    mean_energy_joules                       numeric,
    std_energy_joules                         numeric,
    coefficient_of_variation_energy            numeric,

    computed_at                                 timestamptz not null default now(),

    constraint chk_run_stats_phase check (phase in ('train', 'inference')),
    constraint chk_run_stats_precision check (precision in ('fp32', 'fp16', 'int8', 'int4'))
);

comment on table run_stats_summary is 'Statystyka (mean/std/CV) po repetitions niezaleznych powtorzeniach kazdej unikalnej kombinacji parametrow - patrz docs/measurement_methodology.md';
comment on column run_stats_summary.coefficient_of_variation_energy is 'std_energy_joules / mean_energy_joules - znormalizowana miara rozrzutu, porownywalna miedzy kombinacjami o roznej skali energii';
comment on column run_stats_summary.n_repetitions is 'Liczba powtorzen faktycznie uwzglednionych w statystyce (tylko status=completed z device_id znanym w momencie liczenia) - moze byc mniejsza niz skonfigurowane repetitions, jesli czesc powtorzen sie nie powiodla';

create index idx_run_stats_summary_task_device on run_stats_summary(task, device_id);
create index idx_run_stats_summary_model on run_stats_summary(model_name);
