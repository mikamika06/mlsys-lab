import sys

sys.path.insert(0, ".")
from edgeml.oom import simulate_oom
from edgeml.modelfile import parse_modelfile, verify_modelfile
from edgeml.server_compare import compare_servers

def test_oom_simulation_basic():
    case = {
        "model_size_bytes": 1000,
        "kv_bytes_per_token": 10,
        "memory_limit_bytes": 1500,
        "context_lengths": [10, 100]
    }
    res = simulate_oom(case)
    assert res[0]["oom"] is False
    assert res[1]["oom"] is True

def test_modelfile_parsing():
    text = "FROM test-model-q4_0\nSYSTEM \"hello\"\n"
    parsed = parse_modelfile(text)
    assert parsed["quant"] == "q4_0"
    assert parsed["system"] == "hello"
    assert verify_modelfile(parsed, "hello", "q4_0") is True

def test_server_comparison():
    lm = {"text": "abc", "tokens": 5, "latency_ms": 100.0}
    mlx = {"text": "abc", "tokens": 5, "latency_ms": 90.0}
    res = compare_servers(lm, mlx)
    assert res["text_match"] is True
    assert res["token_match"] is True
    assert res["latency_ratio"] == 0.9
