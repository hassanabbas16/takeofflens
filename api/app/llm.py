"""Thin OpenAI client wrapper: retries, timeouts, token and cost logging.

Deliberately thin. It knows how to call the API, how to record what the call cost, and how
to retry once on a schema failure. It knows nothing about floor plans - that lives in the
pipeline modules.

Two things it does handle that are easy to get wrong:

**Models that reject ``temperature``.** The newest models (gpt-6-astra and later) reject
``temperature``, ``top_p`` and ``top_logprobs`` outright rather than ignoring them. The spec
asks for temperature 0 for determinism, so the wrapper sends it when supported and drops it
when not, rather than hardcoding a model list that goes stale: the first call sends the
parameter, and a rejection is caught, remembered per model, and retried without it. That
costs at most one failed call per model per process.

**Cost is never computed here.** The wrapper records token counts; prices live in
``eval/pricing.yaml`` with the date they were taken. Pricing changes should never require a
code change, and a logged token count stays true whatever the price does later.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Models observed to reject temperature. Populated at runtime from API errors rather than
# hardcoded, so a new model does not need a code change.
_NO_TEMPERATURE: set[str] = set()

_TEMPERATURE_REJECTED_MARKERS = (
    "unsupported parameter",
    "unsupported_parameter",
    "does not support",
    "unrecognized request argument",
    "not supported with this model",
)


class LlmError(RuntimeError):
    """Raised when a call cannot be completed. Callers must catch this - jobs never crash."""


@dataclass
class LlmUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: int = 0
    ok: bool = True
    error: str | None = None
    attempts: int = 1


@dataclass
class LlmResult(Generic[T]):
    parsed: T | None
    usage: LlmUsage

    @property
    def ok(self) -> bool:
        return self.parsed is not None


def _looks_like_temperature_rejection(message: str) -> bool:
    lowered = message.lower()
    return "temperature" in lowered and any(m in lowered for m in _TEMPERATURE_REJECTED_MARKERS)


class LlmClient:
    """Wraps the OpenAI SDK for Structured Outputs calls."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        settings = get_settings()
        self._timeout = timeout if timeout is not None else settings.openai_timeout_seconds
        self._max_retries = settings.openai_max_retries
        if client is not None:
            # Injected for tests; avoids importing the SDK or needing a key.
            self._client = client
            return
        key = api_key or settings.openai_api_key
        if not key or key == "sk-replace-me":
            raise LlmError(
                "OPENAI_API_KEY is not set. Put a real key in .env; "
                ".env.example ships a placeholder on purpose."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=self._timeout)

    def _usage_from_response(self, response: Any, model: str) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached = 0
        details = getattr(usage, "input_tokens_details", None) or getattr(
            usage, "prompt_tokens_details", None
        )
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return int(input_tokens), int(output_tokens), int(cached)

    def _call_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        temperature: float | None,
    ) -> tuple[T | None, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": messages,
            "text_format": schema,
        }
        if temperature is not None and model not in _NO_TEMPERATURE:
            kwargs["temperature"] = temperature

        try:
            response = self._client.responses.parse(**kwargs)
        except Exception as exc:
            message = str(exc)
            if "temperature" in kwargs and _looks_like_temperature_rejection(message):
                # Remember, so every later call for this model skips the parameter.
                _NO_TEMPERATURE.add(model)
                logger.info("model %s rejects temperature; retrying without it", model)
                kwargs.pop("temperature")
                response = self._client.responses.parse(**kwargs)
            else:
                raise

        return getattr(response, "output_parsed", None), response

    def parse(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        temperature: float | None = 0.0,
        purpose: str = "",
    ) -> LlmResult[T]:
        """Call the model and parse into ``schema``.

        Retries once on a schema or transport failure. Never raises for an API problem -
        returns a result with ``parsed=None`` and the error recorded in usage, so a failed
        page marks the job failed rather than taking the process down.
        """
        started = time.perf_counter()
        usage = LlmUsage(model=model)
        last_error: str | None = None

        for attempt in range(1, self._max_retries + 2):
            usage.attempts = attempt
            try:
                parsed, response = self._call_once(
                    model=model, messages=messages, schema=schema, temperature=temperature
                )
                inp, out, cached = self._usage_from_response(response, model)
                # Accumulate: a retry costs real tokens and must show up in the cost report.
                usage.input_tokens += inp
                usage.output_tokens += out
                usage.cached_input_tokens += cached

                if parsed is None:
                    last_error = "model returned no parsed output (refusal or empty)"
                    logger.warning("%s attempt %d: %s", purpose or model, attempt, last_error)
                    continue

                usage.latency_ms = int((time.perf_counter() - started) * 1000)
                usage.ok = True
                return LlmResult(parsed=parsed, usage=usage)

            except ValidationError as exc:
                last_error = f"schema validation failed: {exc}"
                logger.warning("%s attempt %d: %s", purpose or model, attempt, last_error)
            except Exception as exc:  # noqa: BLE001 - recorded, never propagated
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("%s attempt %d: %s", purpose or model, attempt, last_error)

        usage.latency_ms = int((time.perf_counter() - started) * 1000)
        usage.ok = False
        usage.error = last_error
        return LlmResult(parsed=None, usage=usage)


def model_rejects_temperature(model: str) -> bool:
    """Whether a rejection has been observed for this model in this process."""
    return model in _NO_TEMPERATURE


def reset_temperature_cache() -> None:
    """Test hook."""
    _NO_TEMPERATURE.clear()
