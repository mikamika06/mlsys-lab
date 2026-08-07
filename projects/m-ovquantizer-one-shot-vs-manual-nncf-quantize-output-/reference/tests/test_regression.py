import numpy as np
from quantizer.pipeline import manual_nncf_quantize, ov_quantizer_one_shot, verify_output_parity


def test_quantization_parity():
    np.random.seed(42)
    weights = np.random.randn(8, 16).astype(np.float32)
    inputs = np.random.randn(4, 16).astype(np.float32)
    calib = [np.random.randn(4, 16).astype(np.float32) for _ in range(3)]

    config = {
        "nodes": [{"op_type": "MatMul", "version": 14}],
        "graph_opset_map": {"ai.onnx": 14},
        "num_bits": 8,
        "symmetric": True,
        "calibration_data": calib,
    }

    res = verify_output_parity(weights, inputs, config)
    assert res["rel_err"] < 1e-4, f"Relative error {res['rel_err']} exceeds threshold"
    assert res["one_shot_opset"] == res["manual_opset"]


def test_opset_compatibility():
    np.random.seed(42)
    weights = np.random.randn(4, 4).astype(np.float32)
    calib = [np.random.randn(2, 4).astype(np.float32)]

    config = {
        "nodes": [{"op_type": "Conv", "version": 16}],
        "graph_opset_map": {"ai.onnx": 16},
        "num_bits": 8,
        "symmetric": True,
    }

    q1 = ov_quantizer_one_shot(weights, calib, config)
    q2 = manual_nncf_quantize(weights, calib, config)

    assert q1["effective_opset"] == 16
    assert q2["effective_opset"] == 16
    assert np.isclose(q1["w_scale"], q2["w_scale"])
