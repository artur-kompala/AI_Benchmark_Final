-- 0001_init_schema.sql
-- Schemat bazowy dla systemu benchmarkingowego CPU/GPU/NPU.
-- Uruchom w Supabase SQL Editor na swiezym projekcie (patrz docs/setup_supabase.md).

create extension if not exists pgcrypto;

-- ============================================================
-- devices
-- Reprezentuje konkretne urzadzenie obliczeniowe na konkretnej maszynie.
-- Jedna fizyczna maszyna z CPU+GPU to DWA wiersze (rozdzielenie wynikow per urzadzenie).
-- ============================================================
create table devices (
    id              uuid primary key default gen_random_uuid(),
    machine_name    text not null,               -- np. "PC-NVIDIA-RTX4070", "LAPTOP-INTEL-ULTRA7"
    device_type     text not null,                -- 'cpu' | 'cuda' | 'rocm' | 'npu_openvino' | 'npu_directml'
    device_label    text not null,                -- czytelna nazwa, np. "NVIDIA GeForce RTX 4070"
    vendor          text,                          -- 'nvidia' | 'amd' | 'intel'
    cpu_model       text,
    gpu_model       text,
    os_name         text,
    os_version      text,
    driver_version  text,
    total_ram_gb    numeric,
    notes           text,
    created_at      timestamptz not null default now(),

    constraint uq_device unique (machine_name, device_type, device_label),
    constraint chk_device_type check (
        device_type in ('cpu', 'cuda', 'rocm', 'npu_openvino', 'npu_directml')
    )
);

comment on table devices is 'Fizyczna maszyna + konkretne urzadzenie obliczeniowe (CPU/GPU/NPU)';
comment on column devices.device_type is 'cpu | cuda | rocm | npu_openvino | npu_directml';

-- ============================================================
-- runs
-- Jeden przebieg eksperymentu (konkretna kombinacja: task+model+device+batch+precision+phase).
-- ============================================================
create table runs (
    id                       uuid primary key default gen_random_uuid(),
    device_id                uuid not null references devices(id) on delete restrict,

    task                     text not null,        -- 'image_classification' | 'nlp_sentiment' | 'llm_inference'
    model_name               text not null,        -- 'mobilenet_v3' | 'resnet50' | 'distilbert' | ...
    phase                    text not null,        -- 'train' | 'inference'
    dataset                  text,                  -- 'cifar10' | 'imdb' | null dla LLM

    batch_size               integer,
    precision                text not null,         -- 'fp32' | 'fp16' | 'int8' | 'int4'
    num_epochs               integer,               -- null dla inferencji
    num_iterations           integer,               -- liczba powtorzen/iteracji pomiarowych

    config_hash               text not null,         -- sha256 z config_snapshot (idempotencja, wznawianie serii)
    config_snapshot            jsonb not null,        -- pelny zrzut uzytej konfiguracji (audytowalnosc)

    measurement_scope          text not null default 'device_only',  -- 'device_only' | 'npu_only' | 'whole_system'
    warmup_done                 boolean not null default false,

    status                       text not null default 'completed',  -- 'completed' | 'failed' | 'partial'
    error_message                 text,

    started_at                     timestamptz not null,
    finished_at                     timestamptz not null,
    duration_s                       numeric not null,
    throughput_samples_per_s          numeric,

    avg_temperature_c                  numeric,

    created_at                          timestamptz not null default now(),

    constraint chk_run_phase check (phase in ('train', 'inference')),
    constraint chk_run_precision check (precision in ('fp32', 'fp16', 'int8', 'int4')),
    constraint chk_run_measurement_scope check (
        measurement_scope in ('device_only', 'npu_only', 'whole_system')
    ),
    constraint chk_run_status check (status in ('completed', 'failed', 'partial'))
);

comment on column runs.measurement_scope is 'npu_only = zmierzono sam NPU; whole_system = fallback na pobor mocy calego systemu; device_only = CPU/GPU zmierzone bezposrednio';
comment on column runs.config_hash is 'sha256 z config_snapshot, do wykrywania duplikatow / wznawiania macierzy eksperymentow';

-- ============================================================
-- power_samples
-- Surowe probki mocy chwilowej w trakcie przebiegu (~100ms sampling).
-- Wiele zrodel jednoczesnie na jeden run (np. RAPL + codecarbon rownolegle, do wzajemnej walidacji).
-- ============================================================
create table power_samples (
    id            bigint generated always as identity primary key,
    run_id        uuid not null references runs(id) on delete cascade,
    "timestamp"   timestamptz not null,
    source        text not null,                    -- 'rapl' | 'codecarbon' | 'nvml' | 'rocm_smi' | 'npu_native' | 'smart_plug'
    watts         numeric,                            -- moc chwilowa; NULL dopuszczalny gdy odczyt sie nie powiodl, ale rekord logowany
    temperature_c  numeric,                            -- opcjonalnie, jesli zrodlo dostarcza temperature razem z moca

    constraint chk_power_sample_source check (
        source in ('rapl', 'codecarbon', 'nvml', 'rocm_smi', 'npu_native', 'smart_plug')
    )
);

comment on column power_samples.source is 'rapl | codecarbon | nvml | rocm_smi | npu_native | smart_plug';

-- ============================================================
-- run_summary
-- Zagregowane metryki pochodne, relacja 1:1 z runs.
-- ============================================================
create table run_summary (
    run_id                     uuid primary key references runs(id) on delete cascade,

    energy_joules_software      numeric,       -- energia z power_samples software (RAPL/codecarbon/nvml/rocm_smi/npu_native)
    energy_joules_smart_plug     numeric,       -- energia z watomierza fizycznego (manualny lub automatyczny odczyt)
    power_discrepancy_pct         numeric,       -- (|software - smart_plug| / smart_plug) * 100, NULL jesli brak watomierza

    energy_per_sample_joules       numeric,       -- J/sample
    energy_per_epoch_joules         numeric,       -- J/epoch (null dla inferencji)
    avg_power_watts                   numeric,       -- srednia moc chwilowa w przebiegu
    peak_power_watts                   numeric,

    flops_per_sample                    numeric,       -- z thop/fvcore
    flops_per_watt                       numeric,

    samples_processed                     integer,

    computed_at                            timestamptz not null default now()
);

-- ============================================================
-- Indeksy pod typowe zapytania dashboardu
-- ============================================================
create index idx_runs_device_id on runs(device_id);
create index idx_runs_task on runs(task);
create index idx_runs_model_name on runs(model_name);
create index idx_runs_started_at on runs(started_at desc);
create index idx_runs_task_device on runs(task, device_id);      -- porownanie urzadzen per zadanie
create index idx_runs_config_hash on runs(config_hash);          -- wykrywanie duplikatow / wznawianie

create index idx_power_samples_run_id on power_samples(run_id);
create index idx_power_samples_run_ts on power_samples(run_id, "timestamp");
create index idx_power_samples_source on power_samples(source);

-- run_summary.run_id jest juz PK -> osobny indeks niepotrzebny
