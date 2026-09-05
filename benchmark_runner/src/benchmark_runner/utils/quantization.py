"""Dynamiczna kwantyzacja INT8 (PyTorch) dla zadan image_classification i nlp_sentiment -
analogicznie do onnxruntime.quantization uzywanego juz przez llm_inference (patrz
tasks/llm_inference/backends/onnxruntime_backend.py), ale przez torch.quantization.quantize_dynamic
zamiast eksportu do ONNX, zeby nie wymagac dodatkowego kroku eksportu na sciezce pytorch
(CPU/CUDA/ROCm) tych dwoch zadan.

WAZNE OGRANICZENIE: kwantyzacja dynamiczna PyTorch (kernele FBGEMM/QNNPACK) dziala tylko na
CPU - stad wywolujacy (tasks/image_classification/task.py, tasks/nlp_sentiment/task.py)
uzywa tej funkcji wylacznie gdy run_context.device jest typu 'cpu'.

Domyslny mapping dynamicznej kwantyzacji w PyTorch (torch.ao.quantization.quantization_mappings.
get_default_dynamic_quant_module_mappings()) obejmuje nn.Linear (i warstwy rekurencyjne), ale
NIE obejmuje nn.Conv2d - przekazanie nn.Conv2d do quantize_dynamic() nie jest bledem, ale te
warstwy po prostu zostaja w fp32 bez ostrzezenia. Modele oparte glownie o konwolucje
(MobileNetV3, ResNet-50) beda wiec typowo 'partial' (skwantyzowana tylko koncowa warstwa
Linear/classifier), a modele oparte o Linear (DistilBERT) typowo 'success'."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 'success' = wszystkie kwalifikujace sie warstwy (Linear/Conv2d) skwantyzowane,
# 'partial' = tylko czesc (typowo Linear, Conv2d bez wsparcia - patrz docstring modulu),
# 'failed' = kwantyzacja sie nie powiodla lub model nie ma zadnej kwalifikujacej sie warstwy -
#            uruchomienie kontynuuje z oryginalnym (fp32) modelem.
QuantizationStatus = str


def quantize_dynamic_int8(model):
    """Zwraca (model, status). Model wejsciowy powinien byc w trybie eval() - dynamiczna
    kwantyzacja dotyczy tylko inferencji (patrz orchestrator._filter_unsupported_combos,
    ktory pomija kombinacje precision=int8 + phase=train PRZED wywolaniem tej funkcji)."""
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic

    target_types = (nn.Linear, nn.Conv2d)
    # Nazwy (qualified names) warstw kwalifikujacych sie do kwantyzacji W ORYGINALNYM
    # modelu - porownujemy PO NAZWIE (get_submodule), nie przez przeliczenie wszystkich
    # modulow w wyniku: DynamicQuantizedLinear rejestruje wlasny podmodul "_packed_params"
    # (tez z przestrzeni nazw torch.ao.nn.quantized), ktory zawyzalby liczenie "na oko"
    # po samym module.modules() skwantyzowanego modelu.
    target_names = [name for name, m in model.named_modules() if isinstance(m, target_types)]
    total_targets = len(target_names)
    if total_targets == 0:
        logger.warning(
            "Model %s nie zawiera zadnej warstwy Linear/Conv2d do kwantyzacji dynamicznej - "
            "uruchamiam bez zmian (fp32)",
            type(model).__name__,
        )
        return model, "failed"

    try:
        quantized = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    except Exception:
        logger.warning("Dynamiczna kwantyzacja INT8 nie powiodla sie - uruchamiam z fp32", exc_info=True)
        return model, "failed"

    quantized_count = sum(1 for name in target_names if not isinstance(quantized.get_submodule(name), target_types))
    if quantized_count == 0:
        logger.warning(
            "Dynamiczna kwantyzacja INT8 nie skwantyzowala zadnej warstwy (%d kwalifikujacych sie) - "
            "traktuje jako nieudana, model pozostaje fp32",
            total_targets,
        )
        return model, "failed"
    if quantized_count < total_targets:
        logger.info(
            "Dynamiczna kwantyzacja INT8: %d/%d kwalifikujacych sie warstw skwantyzowanych "
            "(reszta - typowo Conv2d, patrz docstring modulu - pozostaje fp32) - status 'partial'",
            quantized_count,
            total_targets,
        )
        return quantized, "partial"

    logger.info("Dynamiczna kwantyzacja INT8: wszystkie %d kwalifikujace sie warstwy skwantyzowane", total_targets)
    return quantized, "success"
