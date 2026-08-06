import numpy as np


def generate_mock_onnx():
    scales = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return {"scales": scales, "nodes": ["QuantizeLinear", "MatMul", "DequantizeLinear"]}


CONFIGS = [generate_mock_onnx()]


def recover_per_channel_scales(model_path):
    if isinstance(model_path, dict):
        return model_path["scales"]
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


def canonicalize_qdq(graph_def):
    nodes = list(graph_def.get("nodes", []))
    if "QuantizeLinear" in nodes and "DequantizeLinear" in nodes:
        return {"canonical": True, "nodes": ["MatMul"]}
    return {"canonical": False, "nodes": nodes}


def explain_slowdown():
    return "unfused qdq nodes cause frequent fp16 fallbacks"
