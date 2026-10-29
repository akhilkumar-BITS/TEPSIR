from app.agent.graph import _parse_verdict


def test_parses_plain_json_verdict():
    raw = (
        '{"verdict": "REVISE", "reason": "Missing receipts", '
        '"next_query": "receipt rules"}'
    )
    verdict = _parse_verdict(raw)

    assert verdict["verdict"] == "REVISE"
    assert verdict["reason"] == "Missing receipts"
    assert verdict["next_query"] == "receipt rules"


def test_parses_json_inside_code_fence():
    raw = '```json\n{"verdict": "PASS", "reason": "Supported", "next_query": null}\n```'
    assert _parse_verdict(raw)["verdict"] == "PASS"


def test_unknown_verdict_falls_back_to_pass():
    assert _parse_verdict('{"verdict": "MAYBE"}')["verdict"] == "PASS"


def test_unparseable_reply_falls_back_to_pass():
    verdict = _parse_verdict("The answer looks fine to me.")
    assert verdict["verdict"] == "PASS"
    assert verdict["next_query"] is None


def test_blank_next_query_becomes_none():
    verdict = _parse_verdict('{"verdict": "REVISE", "reason": "x", "next_query": "  "}')
    assert verdict["next_query"] is None


def test_non_string_next_query_becomes_none():
    verdict = _parse_verdict('{"verdict": "REVISE", "reason": "x", "next_query": 42}')
    assert verdict["next_query"] is None
