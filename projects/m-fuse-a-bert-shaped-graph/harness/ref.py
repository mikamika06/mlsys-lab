import numpy as np

def make_dummy_graph(seed=42):
    rng = np.random.default_rng(seed)
    return {
        "nodes": [
            {"name": "MatMul_Q", "op": "MatMul", "inputs": ["input", "weight_q"]},
            {"name": "Add_Q", "op": "Add", "inputs": ["MatMul_Q", "bias_q"]},
            {"name": "MatMul_K", "op": "MatMul", "inputs": ["input", "weight_k"]},
            {"name": "Add_K", "op": "Add", "inputs": ["MatMul_K", "bias_k"]},
            {"name": "MatMul_V", "op": "MatMul", "inputs": ["input", "weight_v"]},
            {"name": "Add_V", "op": "MatMul_V", "inputs": ["MatMul_V", "bias_v"]},
            {"name": "Attention", "op": "Attention", "inputs": ["Add_Q", "Add_K", "Add_V"]},
            {"name": "LayerNorm", "op": "LayerNorm", "inputs": ["Attention"]},
            {"name": "GeLU", "op": "FastGelu", "inputs": ["LayerNorm"]}
        ],
        "weights": rng.standard_normal((10, 10)).astype(np.float32)
    }

GRAPHS = [make_dummy_graph(i) for i in range(3)]

def fuse_bert_graph(graph):
    nodes = graph["nodes"]
    new_nodes = []
    fused = False
    for n in nodes:
        if n["op"] == "Attention":
            new_nodes.append({"name": "FusedAttention", "op": "FusedAttention", "inputs": n["inputs"]})
            fused = True
        else:
            new_nodes.append(n)
    return {"nodes": new_nodes, "fused": fused}

def triage_attention(graph):
    unfused = [n["name"] for n in graph["nodes"] if n["op"] in ("MatMul", "Add") and "Q" in n["name"]]
    return {"unfused_count": len(unfused), "status": "triaged" if len(unfused) == 0 else "unfused"}

def evaluate_fp16(graph, threshold=1e-3):
    weights = graph["weights"]
    fp16_weights = weights.astype(np.float16).astype(np.float32)
    error = float(np.max(np.abs(weights - fp16_weights)))
    latency_ratio = 0.5
    return {"error": error, "latency_ratio": latency_ratio, "valid": error <= threshold}
