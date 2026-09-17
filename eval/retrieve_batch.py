"""Re-fetch a finished batch's results and cache the raw model output per request.

Retrieving results is **free** - the Batches API bills the inference, not the download, and
results stay available for 29 days. That makes it the right way to answer questions about a
run after the fact (which rooms were hallucinated, what the model actually emitted) without
paying to run it again.

The cached rows written by the Tier 3 run keep only aggregate counts, so anything per-room
has to come from here.

    python eval/retrieve_batch.py --batch-id msgbatch_... --out /eval/cache/batch_raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from app.config import get_settings
from app.llm import LlmClient, LlmError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("/eval/cache/batch_raw"))
    args = parser.parse_args()

    settings = get_settings()
    print(f"api key: {settings.api_key_source}")
    try:
        client = LlmClient()
    except LlmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    raw = client.raw
    batch = raw.messages.batches.retrieve(args.batch_id)
    print(f"batch {batch.id}: {batch.processing_status}")
    print(f"counts: {batch.request_counts}")

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for result in raw.messages.batches.results(args.batch_id):
        custom_id = result.custom_id
        entry: dict = {"custom_id": custom_id, "type": result.result.type}
        message = getattr(result.result, "message", None)
        if message is not None:
            entry["stop_reason"] = message.stop_reason
            entry["model"] = message.model
            entry["usage"] = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
            entry["text"] = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
        else:
            entry["error"] = str(getattr(result.result, "error", "unknown"))
        (args.out / f"{custom_id}.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8"
        )
        written += 1

    print(f"wrote {written} result files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
