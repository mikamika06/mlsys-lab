import numpy as np
from mlxlora.eval_pitfall import LazyNode, build_lazy_lora_graph, measure_execution
from mlxlora.sizes import calculate_model_sizes
from mlxlora.verify import verify_base_weights_untouched

CONFIG_MODEL = {
    "num_layers": 4,
    "bits": 4,
    "group_size": 64,
    "scale_bytes_per_param": 2,
    "non_quantized_bytes_per_param": 2,
    "modules_per_layer": {
        "q_proj": [512, 512],
        "k_proj": [512, 512],
        "v_proj": [512, 512],
        "o_proj": [512, 512],
        "gate_proj": [512, 1408],
        "up_proj": [512, 1408],
        "down_proj": [1408, 512]
    },
    "other_modules": {
        "embed_tokens": [32000, 512],
        "lm_head": [32000, 512],
        "norm": [512]
    }
}

CONFIG_LORA = {
    "r": 16,
    "target_modules": ["q_proj", "v_proj"],
    "adapter_bytes_per_param": 2,
    "safetensors_header_bytes": 1024
}

def generate_sample_graph():
    x = LazyNode("leaf", [], np.array([[1.0, 2.0]], dtype=np.float32))
    w_base = LazyNode("leaf", [], np.array([[0.5, -0.5], [1.0, 0.5]], dtype=np.float32))
    lora_a = LazyNode("leaf", [], np.array([[0.1], [0.2]], dtype=np.float32))
    lora_b = LazyNode("leaf", [], np.array([[0.3, -0.1]], dtype=np.float32))
    scale = 2.0
    root = build_lazy_lora_graph(x, w_base, lora_a, lora_b, scale)
    expected_val = (np.array([[1.0, 2.0]]) @ np.array([[0.5, -0.5], [1.0, 0.5]])) + (
        2.0 * ((np.array([[1.0, 2.0]]) @ np.array([[0.1], [0.2]])) @ np.array([[0.3, -0.1]]))
    )
    return root, expected_val

def generate_sample_weights():
    initial = {
        "model.layers.0.q_proj.weight": np.array([1, 2, 3, 4], dtype=np.uint32),
        "model.layers.0.q_proj.scales": np.array([0.5, 0.5], dtype=np.float16),
        "model.layers.0.v_proj.weight": np.array([5, 6, 7, 8], dtype=np.uint32),
    }
    trained_good = {
        "model.layers.0.q_proj.weight": np.array([1, 2, 3, 4], dtype=np.uint32),
        "model.layers.0.q_proj.scales": np.array([0.5, 0.5], dtype=np.float16),
        "model.layers.0.v_proj.weight": np.array([5, 6, 7, 8], dtype=np.uint32),
        "model.layers.0.q_proj.lora_a": np.array([0.1, 0.2], dtype=np.float32),
    }
    trained_mutated = {
        "model.layers.0.q_proj.weight": np.array([1, 999, 3, 4], dtype=np.uint32),
        "model.layers.0.q_proj.scales": np.array([0.5, 0.5], dtype=np.float16),
        "model.layers.0.v_proj.weight": np.array([5, 6, 7, 8], dtype=np.uint32),
    }
    return initial, trained_good, trained_mutated
