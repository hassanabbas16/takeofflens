"""Thin Anthropic client wrapper: retries, timeouts, token and cost logging.

Deliberately thin. It knows how to call the API, how to record what the call cost, and how
to retry once on a schema failure. It knows nothing about floor plans - that lives in the
pipeline modules.

Structured outputs use ``client.messages.parse(..., output_format=PydanticModel)``, which
constrains the response to the schema and returns a validated instance on
``response.parsed_output``.

Three things it handles that are easy to get wrong:

**Models that reject ``temperature``.** Claude Sonnet 5 and Opus 5 removed the sampling
parameters and return a 400 if you send them; Haiku 4.5 still accepts them. On top of that,
``messages.parse()`` does not expose ``temperature`` at all, so when configured it goes
through ``extra_body``. The wrapper sends it when configured, and on a rejection remembers
that model and retries without it, rather than carrying a hardcoded model list that goes
stale. Costs at most one failed call per model per process.

**``max_tokens`` is required.** Unlike the OpenAI API it is not optional, and hitting it
truncates the extraction mid-room. The cap is configurable, and a ``max_tokens`` stop reason
is treated as a failure worth retrying rather than as a usable result - a truncated room list
is worse than none, because it looks like a complete answer.

**Cost is never computed here.** The wrapper records token counts; prices live in
``eval/pricing.yaml`` with the date they were taken. Anthropic reports cache reads and cache
writes as separate counters that are *not* included in ``input_tokens`` - see app/pricing.py.
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

# Models observed to reject sampling parameters. Populated at runtime from API errors rather
# than hardcoded, so a new model does not need a code change.
_NO_TEMPERATURE: set[str] = set()

_TEMPERATURE_REJECTED_MARKERS = (
    "unsupported parameter",
    "unexpected keyword",
    "does not support",
    "not supported",
    "unrecognized",
    "removed",
)


class LlmError(RuntimeError):
    """Raised when a call cannot be set up. Callers must catch this - jobs never crash."""


@dataclass
class LlmUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Anthropic reports these separately from input_tokens, and prices them differently.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0
    ok: bool = True
    error: str | None = None
    attempts: int = 1
    stop_reason: str | None = None


@dataclass
class LlmResult(Generic[T]):
    parsed: T | None
    usage: LlmUsage

    @property
    def ok(self) -> bool:
        return self.parsed is not None


def _looks_like_parameter_rejection(message: str) -> bool:
    lowered = message.lower()
    return "temperature" in lowered and any(m in lowered for m in _TEMPERATURE_REJECTED_MARKERS)


class LlmClient:
    """Wraps the Anthropic SDK for Structured Outputs calls."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        client: Any = None,
    ) -> None:
        settings = get_settings()
        self._timeout = timeout if timeout is not None else settings.anthropic_timeout_seconds
        self._max_retries = settings.anthropic_max_retries
        self._max_tokens = settings.anthropic_max_tokens
        if client is not None:
            # Injected for tests; avoids importing the SDK or needing a key.
            self._client = client
            return
        key = api_key or settings.anthropic_api_key
        if not key or key.startswith("sk-ant-replace"):
            raise LlmError(
                "ANTHROPIC_API_KEY is not set. Put a real key in .env; "
                ".env.example ships a placeholder on purpose."
            )
        import anthropic

        # max_retries=0: retries are handled here so every attempt's tokens get counted.
        self._client = anthropic.Anthropic(api_key=key, timeout=self._timeout, max_retries=0)

    def _usage_from_response(self, response: Any) -> tuple[int, int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0, 0
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
            int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )

    def _call_once(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[dict[str, Any]],
        schema: type[T],
        temperature: float | None,
    ) -> tuple[T | None, Any, str | None]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "output_format": schema,
        }
        if system:
            kwargs["system"] = system
        # messages.parse() has no temperature parameter, so it rides in extra_body.
        if temperature is not None and model not in _NO_TEMPERATURE:
            kwargs["extra_body"] = {"temperature": temperature}

        try:
            response = self._client.messages.parse(**kwargs)
        except Exception as exc:
            message = str(exc)
            if "extra_body" in kwargs and _looks_like_parameter_rejection(message):
                _NO_TEMPERATURE.add(model)
                logger.info("model %s rejects temperature; retrying without it", model)
                kwargs.pop("extra_body")
                response = self._client.messages.parse(**kwargs)
            else:
                raise

        stop_reason = getattr(response, "stop_reason", None)
        return getattr(response, "parsed_output", None), response, stop_reason

    def parse(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        system: str | None = None,
        temperature: float | None = None,
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
                parsed, response, stop_reason = self._call_once(
                    model=model,
                    system=system,
                    messages=messages,
                    schema=schema,
                    temperature=temperature,
                )
                inp, out, cache_read, cache_write = self._usage_from_response(response)
                # Accumulate: a retry costs real tokens and must show up in the cost report.
                usage.input_tokens += inp
                usage.output_tokens += out
                usage.cache_read_tokens += cache_read
                usage.cache_write_tokens += cache_write
                usage.stop_reason = stop_reason

                if stop_reason == "refusal":
                    last_error = "model declined the request (stop_reason=refusal)"
                    logger.warning("%s attempt %d: %s", purpose or model, attempt, last_error)
                    continue
                if stop_reason == "max_tokens":
                    # A truncated extraction is worse than none - it looks complete.
                    last_error = (
                        f"output hit max_tokens ({self._max_tokens}); "
                        "the extraction would be truncated"
                    )
                    logger.warning("%s attempt %d: %s", purpose or model, attempt, last_error)
                    continue
                if parsed is None:
                    last_error = "model returned no parsed output"
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


    @property
    def raw(self) -> Any:
        """The underlying SDK client.

        Exposed for the Batch API, which lives on the SDK client rather than behind this
        wrapper. Named rather than reached into, so callers are not poking at a private
        attribute and the coupling is visible.
        """
        return self._client


def model_rejects_temperature(model: str) -> bool:
    """Whether a rejection has been observed for this model in this process."""
    return model in _NO_TEMPERATURE


def reset_temperature_cache() -> None:
    """Test hook."""
    _NO_TEMPERATURE.clear()
