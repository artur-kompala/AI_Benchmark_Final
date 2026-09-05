-- 0002_indexes.sql
-- Placeholder na dodatkowe indeksy dodawane w miare rozbudowy dashboardu
-- (np. indeksy czesciowe pod nowe zadania z Fazy 5: nlp_sentiment, llm_inference).
-- Podstawowe indeksy pod runs/power_samples zostaly juz utworzone w 0001_init_schema.sql.

-- Przyklad na przyszlosc (odkomentuj gdy dashboard bedzie tego wymagal):
-- create index idx_run_summary_discrepancy on run_summary(power_discrepancy_pct)
--     where power_discrepancy_pct is not null;
