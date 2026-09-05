-- seed_devices_example.sql
-- Przykladowe wpisy devices do celow developerskich/testowych dashboardu.
-- NIE uruchamiaj na produkcyjnej bazie wynikow - runner sam tworzy wpisy devices
-- (RunRepository.get_or_create_device) na podstawie faktycznie wykrytego sprzetu.

insert into devices (machine_name, device_type, device_label, vendor, cpu_model, gpu_model, os_name, os_version, driver_version, total_ram_gb, notes, gpu_category)
values
    ('DEV-EXAMPLE-PC', 'cpu', 'Example CPU', null, 'Example CPU Model', null, 'Windows 11', '10.0.26200', null, 32, 'Wpis przykladowy - do usuniecia po testach dashboardu', null),
    ('DEV-EXAMPLE-PC', 'cuda', 'Example NVIDIA GPU', 'nvidia', 'Example CPU Model', 'Example NVIDIA GPU Model', 'Windows 11', '10.0.26200', 'CUDA 12.1', 32, 'Wpis przykladowy - do usuniecia po testach dashboardu', 'discrete'),
    ('DEV-EXAMPLE-LAPTOP', 'rocm', 'AMD Radeon(TM) Graphics', 'amd', 'Example Ryzen APU', 'AMD Radeon(TM) Graphics', 'Windows 11', '10.0.26200', 'ROCm 6.1', 16, 'Wpis przykladowy (iGPU) - do usuniecia po testach dashboardu, przyklad devices.gpu_category=integrated', 'integrated')
on conflict (machine_name, device_type, device_label) do nothing;
