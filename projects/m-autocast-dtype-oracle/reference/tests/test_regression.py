from oracle.trace import trace_intermediates
from oracle.diagnose import diagnose_fp32_regions


def test_oracle_fp32_detection():
    sample_graph = {
        "inputs": {"x": "float32", "w": "float32"},
        "ops": [
            {"op": "matmul", "args": ["x", "w"], "out": "h1"},
            {"op": "layernorm", "args": ["h1"], "out": "h2"},
            {"op": "softmax", "args": ["h2"], "out": "out"}
        ]
    }
    res = trace_intermediates(None, sample_graph, autocast_dtype="float16")
    diag = diagnose_fp32_regions(res)
    
    assert len(res["intermediates"]) == 3
    assert res["intermediates"][0]["actual_dtype"] == "float16"
    assert res["intermediates"][1]["actual_dtype"] == "float32"
    assert res["intermediates"][2]["actual_dtype"] == "float32"
    
    reasons = [d["reason"] for d in diag]
    assert "op_requires_fp32" in reasons
