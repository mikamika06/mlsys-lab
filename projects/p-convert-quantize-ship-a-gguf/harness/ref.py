import numpy as np

def create_dummy_hf_model():
    np.random.seed(42)
    return {
        "model.embed_tokens.weight": np.random.randn(100, 64).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(128, 64).astype(np.float32)
    }

def get_dummy_vocab():
    return [f"token_{i}" for i in range(100)]

def get_dummy_dataset():
    np.random.seed(42)
    dataset = []
    for _ in range(3):
        ids = np.random.randint(0, 100, size=16)
        dataset.append({"input_ids": ids, "target_ids": ids})
    return dataset
