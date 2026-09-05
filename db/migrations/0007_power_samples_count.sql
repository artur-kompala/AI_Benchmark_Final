-- 0007_power_samples_count.sql
-- Blad metodologiczny (Poprawka): przy bardzo krotkich przebiegach (np. batch_size=1,
-- duration_s~0.14s) sampler_thread z sampling_interval_ms=100 zdazy zebrac 0-1 probek
-- mocy w calym oknie pomiaru, przez co avg_power_watts to praktycznie pojedynczy losowy
-- odczyt zamiast realnej sredniej (potwierdzone empirycznie: 3 powtorzenia tej samej
-- konfiguracji dawaly 17.6W / 7.3W / 15.4W - rozrzut 2-3x). Runner teraz proaktywnie
-- podnosi num_iterations dla zbyt krotkich przebiegow (patrz core/orchestrator.py
-- _maybe_scale_up_num_iterations), ale liczba faktycznie zebranych probek mocy powinna
-- byc zawsze widoczna jako jawny wskaznik wiarygodnosci avg_power_watts/peak_power_watts
-- w dashboardzie, niezaleznie od tego mechanizmu.

alter table run_summary add column power_samples_count integer;

comment on column run_summary.power_samples_count is 'Liczba probek mocy uzytych do policzenia avg_power_watts/peak_power_watts (po odfiltrowaniu preferred_source) - wskaznik wiarygodnosci tych metryk, zwlaszcza dla bardzo krotkich przebiegow';
