from serving.engine import Engine

def test_determinism():
    e1 = Engine(seed=1337)
    e2 = Engine(seed=1337)
    r1 = e1.generate("the api is good", max_tokens=10)
    r2 = e2.generate("the api is good", max_tokens=10)
    assert r1["tokens"] == r2["tokens"], "Outputs must be identical for same seed"

def test_tokenization_special():
    e = Engine()
    tokens = e.tokenize("the api is good")
    # <s> is 3 in vocab
    assert tokens[0] == 3, "Missing <s> token"

def test_stop_sequences():
    e = Engine(seed=42)
    # </s> is 4
    res = e.generate("hello world", max_tokens=100, stop_tokens=[4])
    assert 4 in res["tokens"], "With 100 tokens and 8 vocab size, 4 should appear"
    assert res["tokens"][-1] == 4, "Stop token must be the last token"
    assert 4 not in res["tokens"][:-1], "Stop token shouldn't appear earlier"

def test_truncation():
    e = Engine(seed=42)
    res = e.generate("hello world", max_tokens=3, stop_tokens=[])
    assert len(res["tokens"]) == 3, f"Expected 3 tokens, got {len(res['tokens'])}"

def test_api_shape():
    e = Engine()
    res = e.generate("hello world", max_tokens=1)
    assert "usage" in res, "Missing usage key"
    assert "prompt_tokens" in res["usage"], "Missing prompt_tokens in usage"
    assert "completion_tokens" in res["usage"], "Missing completion_tokens in usage"
