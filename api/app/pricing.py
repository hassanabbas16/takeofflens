"""Turn logged token counts into a cost figure, using the pricing config file.

Never guesses. A model missing from the table yields ``None``, which the reports render as
"unknown" rather than as zero - a silent zero would understate cost, which is the one
direction that matters here.

Anthropic's token accounting is not the same shape as OpenAI's, and getting it wrong
silently misreports cost: ``input_tokens`` **excludes** cached tokens rather than including
them. Cache reads and cache writes are separate counters with their own rates (a read is
0.1x base input; a 5-minute write is 1.25x). So the total is a sum of four independent
counters, not a subtraction.
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
    output: float
    cache_read: float | None = None
    cache_write: float | None = None


@dataclass(frozen=True)
class PricingTable:
    checked: str
    source: str
    mode: str
    models: dict[str, ModelPrice]

    def cost_usd(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Cost in USD. The four counters are independent and are summed, not netted."""
        price = self.models.get(model)
        if price is None:
            return None
        read_rate = price.cache_read if price.cache_read is not None else price.input
        write_rate = price.cache_write if price.cache_write is not None else price.input
        total = (
            input_tokens * price.input
            + output_tokens * price.output
            + cache_read_tokens * read_rate
            + cache_write_tokens * write_rate
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
            output=float(entry["output"]),
            cache_read=(
                float(entry["cache_read"]) if entry.get("cache_read") is not None else None
            ),
            cache_write=(
                float(entry["cache_write"]) if entry.get("cache_write") is not None else None
            ),
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
