-- 0005_accuracy.sql
-- Dokladnosc klasyfikacji (accuracy) per przebieg - byla juz liczona w
-- TaskStepResult.accuracy (image_classification/nlp_sentiment), ale nigdy nie trafiala
-- do bazy. Potrzebna m.in. do oceny wplywu kwantyzacji INT8 na jakosc modelu (nie tylko
-- na energie) - patrz docs/measurement_methodology.md i dashboard "Wplyw parametrow".

alter table run_summary add column accuracy numeric;

comment on column run_summary.accuracy is 'Dokladnosc klasyfikacji na zbiorze testowym (0-1) - image_classification/nlp_sentiment. NULL dla llm_inference (brak tej metryki) lub gdy nie dalo sie jej policzyc.';
