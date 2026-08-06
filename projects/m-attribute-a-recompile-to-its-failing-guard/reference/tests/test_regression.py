from guardeval.evaluator import evaluate_graph_guards
from guardeval.attribution import Engine


def dummy_compile(meta):
    return {
        "guards": [
            {"type": "shape", "dim": 0, "val": meta["shape"][0]},
            {"type": "dtype", "val": meta["dtype"]},
            {"type": "stride", "dim": 0, "val": meta["strides"][0]}
        ]
    }


def test_stride_guard_attribution():
    meta1 = {"shape": (16, 64), "dtype": "float32", "strides": (64, 1)}
    meta2 = {"shape": (16, 64), "dtype": "float32", "strides": (128, 1)}

    ok, reason = evaluate_graph_guards(dummy_compile(meta1)["guards"], meta2)
    assert not ok, "Stride difference should trigger guard failure"
    assert "strides[0]" in reason, f"Expected stride attribution reason, got: {reason}"

    engine = Engine(dummy_compile)
    res = engine.process_stream([meta1, meta2])
    assert res["recompile_count"] == 2
    assert "strides[0]" in res["attributions"][1]["reason"]
