import sys
sys.path.insert(0, ".")
from lora_pipe.engine import prepare_data, run_lora, merge_adapter, quantize_model, LoraServer, evaluate_quality

def test_prepare_data_output():
    res = prepare_data([{"prompt": "a", "response": "b"}])
    assert len(res) == 1
    assert "tokens" in res[0]
    assert len(res[0]["tokens"]) > 0

def test_run_lora_curve():
    out = run_lora([{"prompt": "a", "response": "b"}], steps=3)
    assert "losses" in out
    assert len(out["losses"]) == 3

def test_merge_adapter():
    base = {"w": __import__("numpy").ones((4, 4))}
    adapter = {"w": {"A": __import__("numpy").zeros((4, 4)), "B": __import__("numpy").zeros((4, 4))}}
    merged = merge_adapter(base, adapter)
    assert "w" in merged
    assert merged["w"].shape == (4, 4)

def test_quantize():
    weights = {"w": __import__("numpy").ones((4, 4))}
    q = quantize_model(weights, bits=4)
    assert "weights" in q
    assert "scales" in q

def test_server():
    server = LoraServer("model")
    res = server.handle_request("test")
    assert isinstance(res, str)
    assert len(res) > 0

def test_evaluate():
    server = LoraServer("model")
    score = evaluate_quality(server, [{"prompt": "test"}])
    assert score >= 0.0
