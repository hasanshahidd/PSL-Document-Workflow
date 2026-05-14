from obs import cache, costs


def test_cache_round_trip_returns_recorded_response():
    key = cache.make_key("model-x", "system text", [{"role": "user", "content": "hi"}], 100)
    assert cache.get(key) is None
    cache.put(key, "model-x", "hello back", input_tokens=5, output_tokens=7)
    hit = cache.get(key)
    assert hit is not None
    assert hit.response_text == "hello back"
    assert (hit.input_tokens, hit.output_tokens) == (5, 7)


def test_cost_record_and_totals_are_consistent():
    usd = costs.record("anthropic", "claude-sonnet-4-6", 1000, 500)
    assert usd > 0
    total = costs.grand_total()
    assert total["calls"] >= 1
    assert total["usd"] >= usd - 1e-9


def test_unknown_model_falls_back_to_default_pricing():
    assert costs.usd_for("unknown-model", 1000, 0) > 0
