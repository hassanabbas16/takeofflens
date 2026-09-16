"""Turn logged token counts into a cost figure, using the pricing config file.

Never guesses. A model missing from the table yields ``None``, which the reports render as
"unknown" rather than as zero - a silent zero would understate cost, which is the one
direction that matters here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import get_settings

PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    input: float
    cached_input: float | None
    output: float


@dataclass(frozen=True)
class PricingTable:
    checked: str
    source: str
    mode: str
    models: dict[str, ModelPrice]

    def cost_usd(
        self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
    ) -> float | None:
        price = self.models.get(model)
        if price is None:
            return None
        # Cached input is billed at the cheaper rate, so it is subtracted from the full-rate
        # input count rather than charged twice.
        cached = min(cached_input_tokens, input_tokens)
        full_rate = input_tokens - cached
        cached_rate = price.cached_input if price.cached_input is not None else price.input
        total = (
            full_rate * price.input
            + cached * cached_rate
            + output_tokens * price.output
        ) / PER_MILLION
        return round(total, 6)

    def knows(self, model: str) -> bool:
        return model in self.models


def load_pricing(path: Path | None = None) -> PricingTable:
    path = Path(path) if path is not None else get_settings().pricing_path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = {
        name: ModelPrice(
            input=float(entry["input"]),
            cached_input=(
                float(entry["cached_input"]) if entry.get("cached_input") is not None else None
            ),
            output=float(entry["output"]),
        )
        for name, entry in (data.get("models") or {}).items()
    }
    return PricingTable(
        checked=str(data.get("checked", "")),
        source=str(data.get("source", "")),
        mode=str(data.get("mode", "standard")),
        models=models,
    )


@lru_cache(maxsize=1)
def get_pricing() -> PricingTable:
    return load_pricing()
