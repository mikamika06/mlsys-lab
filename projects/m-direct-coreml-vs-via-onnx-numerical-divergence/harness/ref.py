import numpy as np

def generate_models():
    np.random.seed(42)
    models = []
    for i in range(3):
        n = 16 + i * 8
        direct_out = np.random.randn(1, n).astype(np.float32)
        noise = 1e-4 * (i + 1) * np.random.randn(1, n).astype(np.float32)
        onnx_out = direct_out + noise
        ops = ["MatMul", "Add", "Softmax", "LayerNorm", "Gelu"]
        node_types = [ops[(i + j) % len(ops)] for j in range(10)]
        providers = ["ANE" if j % 2 == 0 else "CPU" for j in range(10)]
        models.append({
            "id": f"model_{i}",
            "direct": direct_out,
            "onnx": onnx_out,
            "nodes": [{"op": op, "provider": prov} for op, prov in zip(node_types, providers)]
        })
    return models

MODELS = generate_models()

def compute_max_abs_err(model):
    return float(np.max(np.abs(model["direct"] - model["onnx"])))

def compute_census(model):
    counts = {}
    for node in model["nodes"]:
        key = (node["op"], node["provider"])
        counts[key] = counts.get(key, 0) + 1
    return sorted([{"op": k[0], "provider": k[1], "count": v} for k, v in counts.items()], key=lambda x: (x["op"], x["provider"]))

def compute_min_config(models):
    allowed_ops = sorted(list({node["op"] for m in models for node in m["nodes"] if node["provider"] == "ANE"}))
    max_err = max(compute_max_abs_err(m) for m in models)
    return {"allowed_ops": allowed_ops, "max_tolerance": max_err * 1.5}
