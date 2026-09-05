-- 0006_intel_gpu.sql
-- Wsparcie dla Intel GPU (zintegrowane Iris Xe/Arc lub dyskretne Arc A-/B-series) przez
-- OpenVINO GPU plugin - ta sama sciezka wykonania co NPU (patrz devices/intel_gpu.py,
-- power/intel_gpu_power_meter.py), inny device_type/source. Bez tego rozszerzenia
-- devices.device_type='intel_gpu_openvino' i power_samples.source='intel_gpu_native'
-- zostalyby odrzucone przez istniejace check constraints z 0001_init_schema.sql.

alter table devices drop constraint chk_device_type;
alter table devices add constraint chk_device_type check (
    device_type in ('cpu', 'cuda', 'rocm', 'npu_openvino', 'npu_directml', 'intel_gpu_openvino')
);

comment on column devices.device_type is 'cpu | cuda | rocm | npu_openvino | npu_directml | intel_gpu_openvino';

alter table power_samples drop constraint chk_power_sample_source;
alter table power_samples add constraint chk_power_sample_source check (
    source in ('rapl', 'codecarbon', 'nvml', 'rocm_smi', 'npu_native', 'intel_gpu_native', 'smart_plug')
);

comment on column power_samples.source is 'rapl | codecarbon | nvml | rocm_smi | npu_native | intel_gpu_native | smart_plug';
