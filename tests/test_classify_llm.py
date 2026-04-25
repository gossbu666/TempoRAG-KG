"""Test Stage-2 LLM parsing and prompt rendering on a stub client."""
import json
from pathlib import Path
from scripts.classify_failures_llm import render_prompt, parse_response


def test_render_prompt_fills_all_placeholders():
    template = Path("prompts/classify_failure_v1.txt").read_text()
    row = {
        "question": "What is Apple's FY22 revenue?",
        "gold": "$394B",
        "prediction": "$500B",
        "f1": 0.0,
        "retrieved_ids": ["AAPL_FY2022_item7_001"],
    }
    chunks = {"AAPL_FY2022_item7_001":
              {"text": "Apple reported revenue of $394B in fiscal 2022.",
               "ticker": "AAPL", "fy": 2022, "item": "7"}}
    out = render_prompt(template, row, chunks)
    assert "{QUESTION}" not in out
    assert "{GOLD}" not in out
    assert "{PREDICTION}" not in out
    assert "{CONTEXT}" not in out
    assert "{F1}" not in out
    assert "$394B" in out
    assert "AAPL_FY2022_item7_001" in out


def test_parse_response_accepts_clean_json():
    primary, secondary, reason = parse_response(
        '{"primary": "A2", "secondary": null, "reason": "Model invented a number."}'
    )
    assert primary == "A2"
    assert secondary is None
    assert "invented" in reason.lower()


def test_parse_response_strips_fences():
    raw = '```json\n{"primary":"A1","secondary":null,"reason":"miss"}\n```'
    primary, _, _ = parse_response(raw)
    assert primary == "A1"


def test_parse_response_rejects_unknown_code():
    primary, _, _ = parse_response('{"primary":"XX","secondary":null,"reason":"nope"}')
    assert primary is None  # unknown code should fall through
