-- 0008_amd_zenpower_igpu.sql
-- Wsparcie dla dwoch nowych zrodel pomiaru mocy na AMD:
--   1. zenpower (source, patrz power/zenpower_meter.py) - realny pomiar mocy CPU AMD Zen
--      przez modul jadra zenpower3 (telemetria SVI2 z VRM), uzywany na device_type='cpu'
--      obok istniejacych rapl/codecarbon.
--   2. device_type='amd_igpu' (zintegrowane GPU AMD, np. Radeon Vega/780M w APU Ryzen) +
--      jego dwa zrodla: amdgpu_igpu_native (hwmon sterownika amdgpu, patrz
--      power/amdgpu_igpu_power_meter.py) i amd_igpu_whole_system (fallback whole-system
--      jak dla NPU/Intel GPU).
-- Bez tego rozszerzenia zapisy z tych zrodel/tego urzadzenia zostalyby odrzucone przez
-- istniejace check constraints (ten sam problem co przy dodawaniu intel_gpu_openvino w
-- 0006_intel_gpu.sql - patrz tamten komentarz, ten sam blad byl juz raz zglaszany jako
-- "wszystkie zapisy do Supabase koncza sie 400 Bad Request").

alter table devices drop constraint chk_device_type;
alter table devices add constraint chk_device_type check (
    device_type in ('cpu', 'cuda', 'rocm', 'amd_igpu', 'npu_openvino', 'npu_directml', 'intel_gpu_openvino')
);

comment on column devices.device_type is 'cpu | cuda | rocm | amd_igpu | npu_openvino | npu_directml | intel_gpu_openvino';

alter table power_samples drop constraint chk_power_sample_source;
alter table power_samples add constraint chk_power_sample_source check (
    source in (
        'rapl', 'zenpower', 'codecarbon', 'nvml', 'rocm_smi',
        'amdgpu_igpu_native', 'amd_igpu_whole_system',
        'npu_native', 'intel_gpu_native', 'smart_plug'
    )
);

comment on column power_samples.source is 'rapl | zenpower | codecarbon | nvml | rocm_smi | amdgpu_igpu_native | amd_igpu_whole_system | npu_native | intel_gpu_native | smart_plug';
