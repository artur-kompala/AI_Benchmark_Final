"""Zbior promptow testowych do benchmarku inferencji LLM - uzywane cyklicznie
(prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]) az do wyczerpania num_iterations."""

from __future__ import annotations

DEFAULT_PROMPTS: list[str] = [
    "Explain in one sentence what artificial intelligence is.",
    "Write a short poem about autumn.",
    "List three advantages of renewable energy.",
    "Summarize in two sentences what machine learning is.",
    "Briefly describe the difference between a CPU and a GPU.",
]
