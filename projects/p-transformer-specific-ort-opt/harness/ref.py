import numpy as np

def get_sample_graph():
    return {
        "nodes": [
            {"name": "MatMul_Q", "op": "MatMul", "inputs": ["input", "weight_q"], "outputs": ["q"]},
            {"name": "MatMul_K", "op": "MatMul", "inputs": ["input", "weight_k"], "outputs": ["k"]},
            {"name": "MatMul_V", "op": "MatMul", "inputs": ["input", "weight_v"], "outputs": ["v"]},
            {"name": "Attention", "op": "AttentionSubGraph", "inputs": ["q", "k", "v"], "outputs": ["attn_out"]}
        ]
    }

def get_sample_inputs():
    np.random.seed(42)
    return {"input": np.random.randn(1, 128, 768).astype(np.float32)}
