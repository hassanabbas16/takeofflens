"""Tests for the LLM layer, grounding, pairing and pricing.

All of this is exercised with a fake SDK client, so the suite needs no API key and costs
nothing. What is NOT covered here is whether the real models return good extractions - that
is what the Tier 3 cost probe measures, and it cannot be faked.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.llm import (
    LlmClient,
    LlmError,
    model_rejects_temperature,
    reset_temperature_cache,
)
from app.pipeline.extract import encode_page_image
from app.pipeline.grounding import check_grounding, room_bbox_from_tokens
from app.pipeline.ocr import OcrToken
from app.pipeline.pairing import TokenRef, find_candidate_pairs, weighted_distance
from app.pricing import load_pricing
from app.schemas import ExtractedRoom, PlanExtraction, RoomType

# --- fakes -----------------------------------------------------------------------------


class FakeMessages:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        behaviour = self.behaviours.pop(0) if self.behaviours else None
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


class FakeClient:
    def __init__(self, behaviours):
        self.messages = FakeMessages(behaviours)


def response(
    parsed,
    input_tokens=100,
    output_tokens=50,
    cache_read=0,
    cache_write=0,
    stop_reason="end_turn",
):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def room(label="MH", area=11.7, ids=(0, 1), room_type=RoomType.BEDROOM):
    return ExtractedRoom(
        label_raw=label,
        room_type=room_type,
        area_m2=area,
        width_m=None,
        length_m=None,
        source_token_ids=list(ids) if ids is not None else None,
        confidence=0.9,
    )


def refs():
    return [
        TokenRef(id=0, text="MH", confidence=0.95, bbox=(100, 100, 130, 130)),
        TokenRef(id=1, text="11.7", confidence=0.96, bbox=(100, 135, 140, 160)),
        TokenRef(id=2, text="9X21", confidence=0.88, bbox=(400, 400, 450, 425)),
    ]


@pytest.fixture(autouse=True)
def _clear_temperature_cache():
    reset_temperature_cache()
    yield
    reset_temperature_cache()


# --- client behaviour ------------------------------------------------------------------


def test_parse_returns_parsed_and_usage():
    extraction = PlanExtraction(rooms=[room()], notes=None)
    client = LlmClient(client=FakeClient([response(extraction)]))
    result = client.parse(model="m", messages=[], schema=PlanExtraction)
    assert result.ok
    assert result.parsed.rooms[0].label_raw == "MH"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.attempts == 1


def test_retries_once_then_succeeds():
    extraction = PlanExtraction(rooms=[room()], notes=None)
    fake = FakeClient([RuntimeError("transient"), response(extraction)])
    client = LlmClient(client=fake)
    result = client.parse(model="m", messages=[], schema=PlanExtraction)
    assert result.ok
    assert result.usage.attempts == 2


def test_gives_up_after_the_retry_without_raising():
    fake = FakeClient([RuntimeError("boom"), RuntimeError("boom again")])
    client = LlmClient(client=fake)
    result = client.parse(model="m", messages=[], schema=PlanExtraction)
    assert not result.ok
    assert result.parsed is None
    assert result.usage.ok is False
    assert "boom again" in result.usage.error


def test_retry_tokens_are_accumulated_not_replaced():
    """A retry costs real money and must appear in the cost report."""
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([response(None, 80, 10), response(extraction, 100, 50)])
    client = LlmClient(client=fake)
    result = client.parse(model="m", messages=[], schema=PlanExtraction)
    assert result.ok
    assert result.usage.input_tokens == 180
    assert result.usage.output_tokens == 60


def test_temperature_sent_via_extra_body():
    """messages.parse() has no temperature parameter, so it rides in extra_body."""
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([response(extraction)])
    LlmClient(client=fake).parse(
        model="m", messages=[], schema=PlanExtraction, temperature=0.0
    )
    assert fake.messages.calls[0]["extra_body"] == {"temperature": 0.0}


def test_max_tokens_always_sent():
    """The Messages API requires it; omitting it is a 400."""
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([response(extraction)])
    LlmClient(client=fake).parse(model="m", messages=[], schema=PlanExtraction)
    assert fake.messages.calls[0]["max_tokens"] > 0


def test_system_prompt_passed_through():
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([response(extraction)])
    LlmClient(client=fake).parse(
        model="m", messages=[], schema=PlanExtraction, system="rules"
    )
    assert fake.messages.calls[0]["system"] == "rules"


def test_max_tokens_stop_reason_is_a_failure_not_a_result():
    """A truncated room list looks complete, so it must not be accepted."""
    extraction = PlanExtraction(rooms=[room()], notes=None)
    fake = FakeClient([
        response(extraction, stop_reason="max_tokens"),
        response(extraction, stop_reason="max_tokens"),
    ])
    result = LlmClient(client=fake).parse(model="m", messages=[], schema=PlanExtraction)
    assert not result.ok
    assert "max_tokens" in result.usage.error


def test_refusal_stop_reason_is_a_failure():
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([
        response(extraction, stop_reason="refusal"),
        response(extraction, stop_reason="refusal"),
    ])
    result = LlmClient(client=fake).parse(model="m", messages=[], schema=PlanExtraction)
    assert not result.ok
    assert "refusal" in result.usage.error


def test_cache_counters_recorded_separately():
    """Anthropic reports cache reads/writes outside input_tokens; both must be kept."""
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([response(extraction, 100, 50, cache_read=900, cache_write=40)])
    result = LlmClient(client=fake).parse(model="m", messages=[], schema=PlanExtraction)
    assert result.usage.input_tokens == 100
    assert result.usage.cache_read_tokens == 900
    assert result.usage.cache_write_tokens == 40


def test_temperature_rejection_is_detected_and_retried_without_it():
    extraction = PlanExtraction(rooms=[], notes=None)
    rejection = RuntimeError(
        "temperature: Unsupported parameter - sampling parameters were removed on this model."
    )
    fake = FakeClient([rejection, response(extraction)])
    result = LlmClient(client=fake).parse(
        model="claude-sonnet-5", messages=[], schema=PlanExtraction, temperature=0.0
    )
    assert result.ok
    assert "extra_body" in fake.messages.calls[0]
    assert "extra_body" not in fake.messages.calls[1]
    assert model_rejects_temperature("claude-sonnet-5")


def test_temperature_rejection_is_remembered_for_later_calls():
    extraction = PlanExtraction(rooms=[], notes=None)
    rejection = RuntimeError("temperature: Unsupported parameter, it was removed")
    fake = FakeClient([rejection, response(extraction), response(extraction)])
    client = LlmClient(client=fake)
    client.parse(model="claude-sonnet-5", messages=[], schema=PlanExtraction, temperature=0.0)
    client.parse(model="claude-sonnet-5", messages=[], schema=PlanExtraction, temperature=0.0)
    # Third call is the second parse; it must not have re-sent temperature.
    assert "extra_body" not in fake.messages.calls[2]


def test_unrelated_error_is_not_mistaken_for_a_temperature_rejection():
    fake = FakeClient([RuntimeError("rate limit exceeded"), RuntimeError("rate limit")])
    result = LlmClient(client=fake).parse(model="m", messages=[], schema=PlanExtraction)
    assert not result.ok
    assert not model_rejects_temperature("m")


def test_missing_api_key_raises_clearly():
    with pytest.raises(LlmError, match="ANTHROPIC_API_KEY"):
        LlmClient(api_key="sk-ant-replace-me")


# --- grounding -------------------------------------------------------------------------


def test_grounded_room_passes():
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=11.7)], notes=None)
    result = check_grounding(extraction, refs(), hard=True)
    assert result.hallucination_count == 0
    assert len(result.rooms) == 1
    assert not result.dropped


def test_unknown_token_id_is_a_hallucination():
    extraction = PlanExtraction(rooms=[room(ids=(0, 99))], notes=None)
    result = check_grounding(extraction, refs(), hard=True)
    assert result.hallucination_count == 1
    assert result.issues[0].kind == "unknown_token_id"
    assert "99" in result.issues[0].detail


def test_ungrounded_area_is_a_hallucination():
    """An area no OCR token supports - the model invented the number."""
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=99.9)], notes=None)
    result = check_grounding(extraction, refs(), hard=True)
    assert any(i.kind == "ungrounded_area" for i in result.issues)


def test_missing_citations_is_a_hallucination_when_tokens_were_supplied():
    extraction = PlanExtraction(rooms=[room(ids=None, area=11.7)], notes=None)
    result = check_grounding(extraction, refs(), hard=True)
    assert any(i.kind == "missing_token_ids" for i in result.issues)


def test_null_area_is_never_ungrounded():
    extraction = PlanExtraction(rooms=[room(ids=(0,), area=None)], notes=None)
    result = check_grounding(extraction, refs(), hard=True)
    assert result.hallucination_count == 0


def test_area_tolerance_allows_normalisation():
    """The model may return 11.7 for a token reading "11,7"."""
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=11.70001)], notes=None)
    assert check_grounding(extraction, refs(), hard=True).hallucination_count == 0


def test_soft_mode_records_but_does_not_drop():
    """VLM mode: an area OCR missed is reported, not treated as a hallucination to drop."""
    extraction = PlanExtraction(rooms=[room(ids=None, area=42.0)], notes=None)
    result = check_grounding(extraction, refs(), hard=False)
    assert len(result.rooms) == 1
    assert not result.dropped
    # The area disagreement is still recorded as a signal.
    assert any(i.kind == "ungrounded_area" for i in result.issues)


def test_soft_mode_does_not_require_citations():
    extraction = PlanExtraction(rooms=[room(ids=None, area=None)], notes=None)
    result = check_grounding(extraction, refs(), hard=False)
    assert result.hallucination_count == 0


def test_drop_removes_ungrounded_rooms():
    extraction = PlanExtraction(
        rooms=[room(label="MH", ids=(0, 1), area=11.7), room(label="X", ids=(42,), area=11.7)],
        notes=None,
    )
    result = check_grounding(extraction, refs(), hard=True, drop=True)
    assert [r.label_raw for r in result.rooms] == ["MH"]
    assert len(result.dropped) == 1


def test_room_bbox_is_union_of_cited_tokens():
    assert room_bbox_from_tokens(room(ids=(0, 1)), refs()) == [100, 100, 140, 160]


def test_room_bbox_none_without_citations():
    assert room_bbox_from_tokens(room(ids=None), refs()) is None


def test_dimension_pair_token_grounds_an_area():
    token_refs = [TokenRef(id=0, text="3.5 x 4.2", confidence=0.9, bbox=(0, 0, 50, 20))]
    extraction = PlanExtraction(rooms=[room(ids=(0,), area=14.7)], notes=None)
    assert check_grounding(extraction, token_refs, hard=True).hallucination_count == 0


# --- pairing ---------------------------------------------------------------------------


def test_pairs_label_with_nearest_area():
    pairs = find_candidate_pairs(refs())
    assert len(pairs) == 1
    assert pairs[0].label.text == "MH"
    assert pairs[0].area.text == "11.7"
    assert pairs[0].area_m2 == pytest.approx(11.7)


def test_door_code_is_never_offered_as_an_area():
    token_refs = [
        TokenRef(id=0, text="MH", confidence=0.9, bbox=(100, 100, 130, 130)),
        TokenRef(id=1, text="9X21", confidence=0.9, bbox=(100, 135, 140, 160)),
    ]
    assert find_candidate_pairs(token_refs) == []


def test_distant_area_is_not_paired():
    token_refs = [
        TokenRef(id=0, text="MH", confidence=0.9, bbox=(100, 100, 130, 130)),
        TokenRef(id=1, text="11.7", confidence=0.9, bbox=(100, 2000, 140, 2025)),
    ]
    assert find_candidate_pairs(token_refs) == []


def test_vertical_proximity_preferred_over_horizontal():
    label = (100, 100, 130, 130)
    below = (100, 140, 130, 165)
    beside = (172, 100, 202, 130)
    assert weighted_distance(label, below) < weighted_distance(label, beside)


def test_pairing_ignores_implausible_areas():
    token_refs = [
        TokenRef(id=0, text="MH", confidence=0.9, bbox=(100, 100, 130, 130)),
        TokenRef(id=1, text="640 m²", confidence=0.9, bbox=(100, 135, 140, 160)),
    ]
    assert find_candidate_pairs(token_refs) == []


def test_token_refs_use_dense_indices():
    tokens = [
        OcrToken(text="MH", confidence=0.9, bbox=(0, 0, 10, 10), angle=90),
        OcrToken(text="11.7", confidence=0.9, bbox=(0, 12, 10, 22), angle=90),
    ]
    assert [r.id for r in TokenRef.from_tokens(tokens)] == [0, 1]


# --- pricing ---------------------------------------------------------------------------


def test_pricing_loads_and_computes(tmp_path):
    table = load_pricing(_pricing_file(tmp_path))
    # 1M input + 1M output at 1.00 / 5.00
    assert table.cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_cache_counters_are_summed_not_netted(tmp_path):
    """Anthropic excludes cached tokens from input_tokens, so the counters add up.

    Netting them (the OpenAI shape) would undercount by the whole cache read.
    """
    table = load_pricing(_pricing_file(tmp_path))
    cost = table.cost_usd(
        "claude-haiku-4-5",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    # 1.00 base + 0.10 read + 1.25 write
    assert cost == pytest.approx(2.35)


def test_cache_read_is_cheaper_than_base_input(tmp_path):
    table = load_pricing(_pricing_file(tmp_path))
    base = table.cost_usd("claude-haiku-4-5", 1_000_000, 0)
    cached = table.cost_usd("claude-haiku-4-5", 0, 0, cache_read_tokens=1_000_000)
    assert cached < base


def test_unknown_model_cost_is_none_not_zero(tmp_path):
    """A silent zero would understate cost, which is the direction that matters."""
    table = load_pricing(_pricing_file(tmp_path))
    assert table.cost_usd("not-a-model", 1000, 1000) is None
    assert not table.knows("not-a-model")


def test_pricing_records_when_it_was_checked(tmp_path):
    table = load_pricing(_pricing_file(tmp_path))
    assert table.checked
    assert table.source.startswith("http")


def test_shipped_pricing_file_covers_the_default_models():
    """The configured models must be priced, or the cost report says "unknown"."""
    from pathlib import Path

    table = load_pricing(Path(__file__).parents[2] / "eval" / "pricing.yaml")
    assert table.knows("claude-haiku-4-5")
    assert table.knows("claude-sonnet-5")


def _pricing_file(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "checked: '2026-09-17'\n"
        "source: https://platform.claude.com/docs/en/about-claude/pricing\n"
        "mode: standard\n"
        "models:\n"
        "  claude-haiku-4-5:\n"
        "    input: 1.00\n"
        "    output: 5.00\n"
        "    cache_read: 0.10\n"
        "    cache_write: 1.25\n",
        encoding="utf-8",
    )
    return path


# --- image encoding --------------------------------------------------------------------


def test_image_downscaled_to_max_edge():
    image = np.full((1555, 2319, 3), 255, dtype=np.uint8)
    _, width, height = encode_page_image(image, max_px=1536, jpeg_quality=85)
    assert max(width, height) == 1536
    # Aspect ratio preserved.
    assert width / height == pytest.approx(2319 / 1555, rel=0.01)


def test_small_image_not_upscaled():
    image = np.full((400, 600, 3), 255, dtype=np.uint8)
    _, width, height = encode_page_image(image, max_px=1536, jpeg_quality=85)
    assert (width, height) == (600, 400)


def test_encoded_image_is_base64_jpeg():
    import base64

    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    encoded, _, _ = encode_page_image(image, max_px=1536, jpeg_quality=85)
    raw = base64.b64decode(encoded)
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker


# --- schema ----------------------------------------------------------------------------


def test_room_type_enum_covers_the_lexicon_and_other():
    values = {m.value for m in RoomType}
    assert "bedroom" in values
    assert "kitchen" in values
    assert "other" in values


def test_schema_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ExtractedRoom(
            label_raw="MH", room_type=RoomType.BEDROOM, area_m2=1.0,
            width_m=None, length_m=None, source_token_ids=None, confidence=0.5,
            surprise="nope",
        )


def test_schema_is_json_schema_serialisable():
    """The schema is what constrains the model, so it must generate cleanly."""
    schema = PlanExtraction.model_json_schema()
    room_schema = schema["$defs"]["ExtractedRoom"]
    # extra="forbid" is what produces additionalProperties: false.
    assert room_schema["additionalProperties"] is False
    # Nullable fields are required-but-nullable, so the model must state "no area here"
    # rather than omitting the key - grounding depends on that distinction.
    assert set(room_schema["required"]) == set(room_schema["properties"])
    assert {"type": "null"} in room_schema["properties"]["area_m2"]["anyOf"]


def test_pydantic_model_accepted_by_the_sdk_as_an_output_format():
    """Guards the structured-outputs entry point against an SDK shape change."""
    import inspect

    import anthropic

    params = inspect.signature(anthropic.Anthropic(api_key="sk-ant-x").messages.parse).parameters
    assert "output_format" in params
    assert "max_tokens" in params


# --- batch API -------------------------------------------------------------------------


def test_batch_discount_is_half_of_standard(tmp_path):
    table = load_pricing(_pricing_file(tmp_path))
    standard = table.cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000)
    batched = table.cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000, batch=True)
    assert batched == pytest.approx(standard / 2)


def test_batch_request_carries_a_json_schema_not_the_parse_helper():
    """Batch requests take raw params, so structured output goes via output_config."""
    from app.batch import BatchItem, build_request

    item = BatchItem(
        custom_id="plan-1191",
        model="claude-haiku-4-5",
        system="rules",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8000,
    )
    request = build_request(item)
    assert request["custom_id"] == "plan-1191"
    params = request["params"]
    assert params["max_tokens"] == 8000
    assert params["system"] == "rules"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "rooms" in params["output_config"]["format"]["schema"]["properties"]


def test_batch_results_are_matched_by_custom_id_not_position():
    """Results come back in any order; positional matching would mix up plans."""
    from app.batch import BatchItem, submit_and_wait

    extraction = PlanExtraction(rooms=[room(label="MH")], notes=None)
    other = PlanExtraction(rooms=[room(label="OH")], notes=None)

    def message_for(payload):
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=payload.model_dump_json())],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            ),
        )

    class FakeBatches:
        def create(self, requests):
            return SimpleNamespace(id="batch_1")

        def retrieve(self, batch_id):
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id):
            # Deliberately reversed relative to the submitted order.
            return [
                SimpleNamespace(custom_id="b", result=SimpleNamespace(
                    type="succeeded", message=message_for(other))),
                SimpleNamespace(custom_id="a", result=SimpleNamespace(
                    type="succeeded", message=message_for(extraction))),
            ]

    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
    items = [
        BatchItem("a", "claude-haiku-4-5", None, [{"role": "user", "content": "x"}], 100),
        BatchItem("b", "claude-haiku-4-5", None, [{"role": "user", "content": "y"}], 100),
    ]
    outcomes = submit_and_wait(client, items, poll_seconds=0)

    assert outcomes["a"].parsed.rooms[0].label_raw == "MH"
    assert outcomes["b"].parsed.rooms[0].label_raw == "OH"


def test_batch_missing_result_is_reported_not_silently_dropped():
    from app.batch import BatchItem, submit_and_wait

    class FakeBatches:
        def create(self, requests):
            return SimpleNamespace(id="batch_1")

        def retrieve(self, batch_id):
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id):
            return []

    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
    items = [BatchItem("a", "m", None, [{"role": "user", "content": "x"}], 100)]
    outcomes = submit_and_wait(client, items, poll_seconds=0)
    assert not outcomes["a"].ok
    assert "no result" in outcomes["a"].error


def test_batch_max_tokens_stop_reason_is_a_failure():
    from app.batch import BatchItem, submit_and_wait

    class FakeBatches:
        def create(self, requests):
            return SimpleNamespace(id="b1")

        def retrieve(self, batch_id):
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id):
            return [SimpleNamespace(custom_id="a", result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(
                    stop_reason="max_tokens", content=[],
                    usage=SimpleNamespace(
                        input_tokens=1, output_tokens=1,
                        cache_read_input_tokens=0, cache_creation_input_tokens=0),
                ),
            ))]

    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))
    items = [BatchItem("a", "m", None, [{"role": "user", "content": "x"}], 100)]
    outcomes = submit_and_wait(client, items, poll_seconds=0)
    assert not outcomes["a"].ok
    assert "max_tokens" in outcomes["a"].error


# --- api key source --------------------------------------------------------------------


def test_api_key_is_read_from_the_env_file_not_the_environment(tmp_path, monkeypatch):
    """An ambient ANTHROPIC_API_KEY must never pay for this project's calls."""
    from app.config import read_env_file_value

    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")

    assert read_env_file_value("ANTHROPIC_API_KEY", env_file) == "sk-ant-from-file"


def test_missing_env_file_yields_empty_not_the_environment(tmp_path, monkeypatch):
    from app.config import read_env_file_value

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
    assert read_env_file_value("ANTHROPIC_API_KEY", tmp_path / "absent.env") == ""


def test_env_file_values_may_be_quoted(tmp_path):
    from app.config import read_env_file_value

    env_file = tmp_path / ".env"
    env_file.write_text('ANTHROPIC_API_KEY="sk-ant-quoted"\n', encoding="utf-8")
    assert read_env_file_value("ANTHROPIC_API_KEY", env_file) == "sk-ant-quoted"


def test_env_file_comments_and_blanks_ignored(tmp_path):
    from app.config import read_env_file_value

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# ANTHROPIC_API_KEY=sk-ant-commented-out\n\nANTHROPIC_API_KEY=sk-ant-real\n",
        encoding="utf-8",
    )
    assert read_env_file_value("ANTHROPIC_API_KEY", env_file) == "sk-ant-real"


def test_custom_id_encodes_the_plus_in_ocr_llm():
    """The Batch API rejects a whole batch over one bad custom_id, and "ocr+llm" has a "+".

    A 150-request batch was rejected with a 400 naming only "requests.0" because the
    natural key "1191|ocr+llm" is not a legal id.
    """
    from app.batch import CUSTOM_ID_RE, safe_custom_id

    encoded = safe_custom_id("1191|ocr+llm")
    assert CUSTOM_ID_RE.match(encoded), encoded


def test_custom_id_encoding_does_not_collide():
    """"ocr+llm" and "ocr_llm" both sanitise to the same string; they must not collide."""
    from app.batch import safe_custom_id

    assert safe_custom_id("1191|ocr+llm") != safe_custom_id("1191|ocr_llm")


def test_custom_id_leaves_already_legal_ids_alone():
    from app.batch import safe_custom_id

    assert safe_custom_id("1191_rules") == "1191_rules"


def test_custom_id_stays_within_the_length_limit():
    from app.batch import CUSTOM_ID_RE, safe_custom_id

    encoded = safe_custom_id("x" * 200)
    assert len(encoded) <= 64
    assert CUSTOM_ID_RE.match(encoded)


def test_submitting_an_illegal_custom_id_fails_before_the_api_call():
    """Fail locally, naming the id, rather than paying a round trip to be told "requests.0"."""
    from app.batch import BatchItem, submit_and_wait

    class Boom:
        class messages:
            class batches:
                @staticmethod
                def create(**_kwargs):
                    raise AssertionError("must not reach the API")

    item = BatchItem(
        custom_id="1191|ocr+llm",
        model="claude-haiku-4-5",
        system=None,
        messages=[],
        max_tokens=1024,
    )
    with pytest.raises(ValueError, match="custom_id"):
        submit_and_wait(Boom(), [item])


def test_duplicate_custom_ids_are_refused():
    """Results are matched by id, so duplicates would misattribute one plan's rooms."""
    from app.batch import BatchItem, submit_and_wait

    def item(custom_id):
        return BatchItem(custom_id=custom_id, model="m", system=None, messages=[], max_tokens=8)

    with pytest.raises(ValueError, match="unique"):
        submit_and_wait(object(), [item("same"), item("same")])


# --- grounding against garbled OCR -------------------------------------------------------
#
# All of these token strings are real reads from the 50-plan Tier 3 run. PaddleOCR renders
# the superscript in "m²" as LaTeX-like noise, which the parser cannot read as an area.
# Before this was handled, a model that read such a token *correctly* was scored as
# hallucinating: 38 of ocr+llm's 39 failures and 47 of hybrid's 57.


def _garbled_refs(text):
    return [
        TokenRef(id=0, text="MH", confidence=0.95, bbox=(100, 100, 130, 130)),
        TokenRef(id=1, text=text, confidence=0.70, bbox=(100, 135, 140, 160)),
    ]


@pytest.mark.parametrize(
    ("text", "area"),
    [
        ("3,3 m^{2}$", 3.3),
        ("6,7 m^{2}$", 6.7),
        ("15.5 m^2}$", 15.5),
        ("9,5m^2}$", 9.5),
    ],
)
def test_a_repaired_area_unit_grounds_strictly(text, area):
    """normalise_area_unit repairs the mangled superscript, so the parser now reads these.

    They ground on the strict path and raise no soft note at all - the rescue below is no
    longer needed for them. Kept as a regression: if the repair stops working, these fall
    back to the soft path and this test says so.
    """
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=area)], notes=None)
    result = check_grounding(extraction, _garbled_refs(text), hard=True)
    assert result.hallucination_count == 0
    assert not any(i.kind == "area_from_unparsed_token" for i in result.issues)
    assert len(result.rooms) == 1


@pytest.mark.parametrize(
    ("text", "area"),
    [
        ("OH. 16.0 n?", 16.0),   # "m²" misread as "n?" - not superscript debris
        ("OLOH. 17.6m", 17.6),   # area printed with what looks like a length unit
        ("MH18,4m", 18.4),       # no space between label and number
    ],
)
def test_area_read_from_a_still_unparseable_token_is_not_a_hallucination(text, area):
    """Corruptions the repair deliberately does not touch, rescued by grounding instead."""
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=area)], notes=None)
    result = check_grounding(extraction, _garbled_refs(text), hard=True)
    assert result.hallucination_count == 0
    assert any(i.kind == "area_from_unparsed_token" for i in result.issues)
    # Still a real room: a soft note is not a verdict.
    assert len(result.rooms) == 1
    assert not result.dropped


def test_area_absent_from_the_page_is_still_a_hallucination():
    """The rescue must not become a blanket pass."""
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=42.0)], notes=None)
    result = check_grounding(extraction, _garbled_refs("3,3 m^{2}$"), hard=True)
    assert result.hallucination_count == 1
    assert any(i.kind == "ungrounded_area" for i in result.issues)


def test_a_bare_integer_in_a_door_code_does_not_ground_an_area():
    """"9X21" contains 21. A model claiming 21 m2 has not transcribed an area."""
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=21.0)], notes=None)
    result = check_grounding(extraction, refs(), hard=True)
    assert result.hallucination_count == 1
    assert any(i.kind == "ungrounded_area" for i in result.issues)


def test_soft_note_does_not_mark_the_room_ungrounded():
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=3.3)], notes=None)
    result = check_grounding(extraction, _garbled_refs("3,3 m^{2}$"), hard=True)
    assert [r.label_raw for r in result.rooms] == ["MH"]
    assert result.dropped == []


@pytest.mark.parametrize(
    ("text", "area"),
    [
        ("0LOH.28m2$", 28.0),   # "OLOH. 28 m2", integer area with a damaged unit
        ("65 m^{2", 65.0),
    ],
)
def test_integer_area_with_a_damaged_unit_is_transcribed_not_invented(text, area):
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=area)], notes=None)
    result = check_grounding(extraction, _garbled_refs(text), hard=True)
    assert result.hallucination_count == 0


def test_a_bare_integer_with_no_unit_still_does_not_ground_an_area():
    """"85m" could be a length. Only a unit corroborates an integer."""
    extraction = PlanExtraction(rooms=[room(ids=(0, 1), area=85.0)], notes=None)
    result = check_grounding(extraction, _garbled_refs("LH+K 85m"), hard=True)
    assert result.hallucination_count == 1
