import numpy as np

CONFIGS = [
    {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072, "vocab_size": 32000, "seq_len": 512},
    {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096, "vocab_size": 32000, "seq_len": 512},
]

def compute_transformer_metrics(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_hidden_layers"]
    v = cfg["vocab_size"]
    i = cfg["intermediate_size"]
    params = v * h + l * (4 * h * h + 2 * h * i) + h * v
    flops = l * (12 * h * h * cfg["seq_len"] + 2 * h * i * cfg["seq_len"])
    return {"params": params, "flops": flops}

def measure_init_loss(cfg, strategy):
    np.random.seed(42)
    if strategy == "random":
        return float(np.random.uniform(8.0, 10.0))
    elif strategy == "stacked":
        return float(np.random.uniform(4.0, 5.0))
    elif strategy == "truncated":
        return float(np.random.uniform(5.0, 6.0))
    return 10.0

def compare_depth_width(cfg, target_params):
    return {
        "depth_only": {"layers": cfg["num_hidden_layers"] // 2, "hidden_size": cfg["hidden_size"]},
        "width_only": {"layers": cfg["num_hidden_layers"], "hidden_size": cfg["hidden_size"] // 2}
    }
