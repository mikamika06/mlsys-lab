import numpy as np
from edgeonnx.export import parse_export_config, simulate_export
from edgeonnx.runner import run_inference, measure_latency
from edgeonnx.inspect import get_node_placement, find_fallback_nodes


def test_export_and_inference():
    cfg = {"model_name": "test-model", "hidden_size": 32, "num_layers": 1, "vocab_size": 128}
    parsed = parse_export_config(cfg)
    spec = simulate_export(parsed)
    inputs = np.array([1, 2, 3], dtype=np.float32)
    out = run_inference(spec, inputs, "CoreMLExecutionProvider")
    assert out.shape == inputs.shape


def test_fallback_detection():
    spec = {
        "nodes": [
            {"name": "Op1", "provider": "CoreMLExecutionProvider"},
            {"name": "Op2", "provider": "CPUExecutionProvider"}
        ]
    }
    fallbacks = find_fallback_nodes(spec, "CoreMLExecutionProvider")
    assert "Op2" in fallbacks
    assert "Op1" not in fallbacks


def test_latency_comparison():
    spec = {"nodes": [{"name": "Op1", "provider": "CoreMLExecutionProvider"}]}
    inputs = np.array([1.0], dtype=np.float32)
    lat_coreml = measure_latency(spec, inputs, "CoreMLExecutionProvider")
    lat_cpu = measure_latency(spec, inputs, "CPUExecutionProvider")
    assert lat_coreml > 0
    assert lat_cpu > 0
