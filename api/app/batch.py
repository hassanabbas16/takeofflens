"""Message Batches: the same three approaches at half price, asynchronously.

The Batch API charges 50% of standard rates on both input and output, which is the largest
single cost lever available for a run over many plans. The trade is latency - a batch is
processed asynchronously and may take minutes - which is irrelevant for an eval sweep and
unacceptable for the interactive upload path, so batching lives here rather than in the
request path.

Two differences from the synchronous client that matter:

**No ``messages.parse`` helper.** A batch request carries raw parameters, so structured
output is requested with ``output_config.format`` (a JSON schema) and the response text is
validated against the Pydantic model here. Same schema, same guarantees, more plumbing.

**Results come back in any order and keyed by ``custom_id``.** They are matched by that id
and never by position - positional matching is the classic way to silently attribute one
plan's rooms to another.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm import LlmUsage
from app.schemas import PlanExtraction

logger = logging.getLogger(__name__)

# How often to ask whether the batch has finished, and how long to wait before giving up.
POLL_SECONDS = 15
DEFAULT_TIMEOUT_SECONDS = 60 * 60

# The Batch API accepts only these characters in a custom_id, and rejects the whole batch
# with a 400 if any one id breaks the rule. Our natural key for a request is
# (plan, approach), and "ocr+llm" contains a "+" - so the natural key is *not* a legal id
# and has to be encoded. This bit the first real submission of a 150-request batch.
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_CUSTOM_ID_MAX = 64


def safe_custom_id(raw: str) -> str:
    """Encode an arbitrary key as a legal custom_id, injectively.

    Illegal characters become "_". That alone is not injective - "ocr+llm" and "ocr_llm"
    would collide - so whenever the encoding changes the string or has to be truncated, a
    short digest of the original is appended. Ids are only ever used to look a result back
    up, never parsed apart, so an opaque suffix costs nothing and a collision would
    silently attribute one plan's rooms to another.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if cleaned == raw and len(cleaned) <= _CUSTOM_ID_MAX:
        return cleaned
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: _CUSTOM_ID_MAX - 9]}-{digest}"


@dataclass
class BatchItem:
    """One request in a batch, tagged so its result can be found again."""

    custom_id: str
    model: str
    system: str | None
    messages: list[dict[str, Any]]
    max_tokens: int


@dataclass
class BatchOutcome:
    custom_id: str
    parsed: PlanExtraction | None
    usage: LlmUsage
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.parsed is not None


def _json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "schema": schema.model_json_schema(),
        }
    }


def build_request(item: BatchItem, schema: type[BaseModel] = PlanExtraction) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": item.model,
        "max_tokens": item.max_tokens,
        "messages": item.messages,
        "output_config": _json_schema_format(schema),
    }
    if item.system:
        params["system"] = item.system
    return {"custom_id": item.custom_id, "params": params}


def _usage_from(message: Any, model: str) -> LlmUsage:
    usage = getattr(message, "usage", None)
    return LlmUsage(
        model=model,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0) if usage else 0,
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0) if usage else 0,
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0,
        cache_write_tokens=(
            int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        ),
        stop_reason=getattr(message, "stop_reason", None),
    )


def _parse_message(message: Any, model: str, schema: type[BaseModel]) -> BatchOutcome:
    usage = _usage_from(message, model)
    custom_id = ""

    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        usage.ok = False
        usage.error = "output hit max_tokens; the extraction would be truncated"
        return BatchOutcome(custom_id, None, usage, usage.error)
    if stop_reason == "refusal":
        usage.ok = False
        usage.error = "model declined the request (stop_reason=refusal)"
        return BatchOutcome(custom_id, None, usage, usage.error)

    text = next(
        (b.text for b in getattr(message, "content", []) if getattr(b, "type", "") == "text"),
        None,
    )
    if not text:
        usage.ok = False
        usage.error = "no text block in the batch response"
        return BatchOutcome(custom_id, None, usage, usage.error)

    try:
        parsed = schema.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        usage.ok = False
        usage.error = f"could not validate batch response against the schema: {exc}"
        return BatchOutcome(custom_id, None, usage, usage.error)

    return BatchOutcome(custom_id, parsed, usage)  # type: ignore[arg-type]


def submit_and_wait(
    client: Any,
    items: list[BatchItem],
    *,
    schema: type[BaseModel] = PlanExtraction,
    poll_seconds: int = POLL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    on_progress: Any = None,
) -> dict[str, BatchOutcome]:
    """Submit one batch, wait for it, and return outcomes keyed by ``custom_id``."""
    if not items:
        return {}

    # Fail here, with the offending id named, rather than letting the API reject all 150
    # requests with a message that identifies only "requests.0".
    illegal = [i.custom_id for i in items if not CUSTOM_ID_RE.match(i.custom_id)]
    if illegal:
        raise ValueError(
            f"{len(illegal)} custom_id(s) are not accepted by the Batch API "
            f"(must match {CUSTOM_ID_RE.pattern}); first: {illegal[0]!r}. "
            "Build them with safe_custom_id()."
        )
    seen = {i.custom_id for i in items}
    if len(seen) != len(items):
        raise ValueError(
            "custom_ids must be unique within a batch; results are matched by id and "
            "duplicates would attribute one request's result to another"
        )

    requests = [build_request(item, schema) for item in items]
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info("submitted batch %s with %d requests", batch_id, len(requests))

    started = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if time.time() - started > timeout_seconds:
            raise TimeoutError(
                f"batch {batch_id} still {batch.processing_status} after "
                f"{timeout_seconds}s; it is not lost - retrieve it by id later"
            )
        if on_progress is not None:
            on_progress(batch)
        time.sleep(poll_seconds)

    models = {item.custom_id: item.model for item in items}
    outcomes: dict[str, BatchOutcome] = {}

    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        model = models.get(custom_id, "")
        result = entry.result
        if result.type != "succeeded":
            detail = getattr(result, "error", None)
            usage = LlmUsage(model=model, ok=False, error=f"batch result {result.type}: {detail}")
            outcomes[custom_id] = BatchOutcome(custom_id, None, usage, usage.error)
            continue
        outcome = _parse_message(result.message, model, schema)
        outcomes[custom_id] = BatchOutcome(
            custom_id, outcome.parsed, outcome.usage, outcome.error
        )

    missing = [item.custom_id for item in items if item.custom_id not in outcomes]
    for custom_id in missing:
        usage = LlmUsage(model=models.get(custom_id, ""), ok=False, error="no result returned")
        outcomes[custom_id] = BatchOutcome(custom_id, None, usage, usage.error)

    return outcomes
