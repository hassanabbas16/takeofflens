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


class FakeResponses:
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
        self.responses = FakeResponses(behaviours)


def response(parsed, input_tokens=100, output_tokens=50, cached=0):
    return SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
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


def test_temperature_sent_by_default():
    extraction = PlanExtraction(rooms=[], notes=None)
    fake = FakeClient([response(extraction)])
    LlmClient(client=fake).parse(
        model="m", messages=[], schema=PlanExtraction, temperature=0.0
    )
    assert fake.responses.calls[0]["temperature"] == 0.0


def test_temperature_rejection_is_detected_and_retried_without_it():
    extraction = PlanExtraction(rooms=[], notes=None)
    rejection = RuntimeError(
        "Unsupported parameter: 'temperature' is not supported with this model."
    )
    fake = FakeClient([rejection, response(extraction)])
    result = LlmClient(client=fake).parse(
        model="gpt-6-astra", messages=[], schema=PlanExtraction, temperature=0.0
    )
    assert result.ok
    assert "temperature" in fake.responses.calls[0]
    assert "temperature" not in fake.responses.calls[1]
    assert model_rejects_temperature("gpt-6-astra")


def test_temperature_rejection_is_remembered_for_later_calls():
    extraction = PlanExtraction(rooms=[], notes=None)
    rejection = RuntimeError("Unsupported parameter: 'temperature' is not supported")
    fake = FakeClient([rejection, response(extraction), response(extraction)])
    client = LlmClient(client=fake)
    client.parse(model="gpt-6-astra", messages=[], schema=PlanExtraction, temperature=0.0)
    client.parse(model="gpt-6-astra", messages=[], schema=PlanExtraction, temperature=0.0)
    # Third call is the second parse; it must not have re-sent temperature.
    assert "temperature" not in fake.responses.calls[2]


def test_unrelated_error_is_not_mistaken_for_a_temperature_rejection():
    fake = FakeClient([RuntimeError("rate limit exceeded"), RuntimeError("rate limit")])
    result = LlmClient(client=fake).parse(model="m", messages=[], schema=PlanExtraction)
    assert not result.ok
    assert not model_rejects_temperature("m")


def test_missing_api_key_raises_clearly():
    with pytest.raises(LlmError, match="OPENAI_API_KEY"):
        LlmClient(api_key="sk-replace-me")


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
    # 1M input + 1M output at 2.00 / 12.00
    assert table.cost_usd("gpt-5.6-terra", 1_000_000, 1_000_000) == pytest.approx(14.0)


def test_cached_input_billed_at_the_cheaper_rate(tmp_path):
    table = load_pricing(_pricing_file(tmp_path))
    # 1M input of which 1M cached, at 0.20, plus no output.
    assert table.cost_usd("gpt-5.6-terra", 1_000_000, 0, 1_000_000) == pytest.approx(0.20)


def test_unknown_model_cost_is_none_not_zero(tmp_path):
    """A silent zero would understate cost, which is the direction that matters."""
    table = load_pricing(_pricing_file(tmp_path))
    assert table.cost_usd("not-a-model", 1000, 1000) is None
    assert not table.knows("not-a-model")


def test_pricing_records_when_it_was_checked(tmp_path):
    table = load_pricing(_pricing_file(tmp_path))
    assert table.checked
    assert table.source.startswith("http")


def _pricing_file(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "checked: '2026-09-17'\n"
        "source: https://developers.openai.com/api/docs/pricing\n"
        "mode: standard\n"
        "models:\n"
        "  gpt-5.6-terra:\n"
        "    input: 2.00\n"
        "    cached_input: 0.20\n"
        "    output: 12.00\n",
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


def test_strict_json_schema_is_generated():
    """Structured Outputs runs in strict mode; the schema must survive conversion."""
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(PlanExtraction)
    room_schema = schema["$defs"]["ExtractedRoom"]
    assert room_schema["additionalProperties"] is False
    # Strict mode requires every property to be required, nullability via anyOf.
    assert set(room_schema["required"]) == set(room_schema["properties"])
