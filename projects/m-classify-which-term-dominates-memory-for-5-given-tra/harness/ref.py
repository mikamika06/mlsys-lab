import numpy as np

CONFIGS = [
    {"hidden_size": 256, "intermediate_size": 1024, "num_heads": 4, "seq_len": 128, "batch_size": 2},
    {"hidden_size": 512, "intermediate_size": 2048, "num_heads": 8, "seq_len": 2048, "batch_size": 1},
    {"hidden_size": 128, "intermediate_size": 256, "num_heads": 2, "seq_len": 64, "batch_size": 1},
    {"hidden_size": 256, "intermediate_size": 1024, "num_heads": 4, "seq_len": 1024, "batch_size": 2},
    {"hidden_size": 512, "intermediate_size": 4096, "num_heads": 8, "seq_len": 512, "batch_size": 1},
]

def classify_dominating_term(config):
    b = config["batch_size"]
    s = config["seq_len"]
    h = config["hidden_size"]
    i = config["intermediate_size"]

    attn_term = b * s * s * config["num_heads"]
    mlp_term = b * s * i
    hidden_term = b * s * h

    terms = {
        "attention_matrix": attn_term,
        "mlp_intermediate": mlp_term,
        "hidden_states": hidden_term
    }
    return max(terms, key=terms.get)

def measure_peak_memory(config):
    b = config["batch_size"]
    s = config["seq_len"]
    h = config["hidden_size"]
    i = config["intermediate_size"]

    bytes_per_elem = 4
    attn = b * s * s * config["num_heads"] * bytes_per_elem
    mlp = b * s * i * bytes_per_elem
    hid = b * s * h * bytes_per_elem * 4
    return float(attn + mlp + hid)

def measure_scaling(hidden_size, intermediate_size, num_heads, batch_size):
    lengths = [64, 128, 256, 512]
    vals = []
    for s in lengths:
        cfg = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_heads": num_heads,
            "seq_len": s,
            "batch_size": batch_size
        }
        vals.append(measure_peak_memory(cfg))
    return vals
