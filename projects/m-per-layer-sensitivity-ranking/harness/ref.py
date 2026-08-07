import numpy as np

CONFIGS = [
    {
        "layers": [
            {"index": 0, "size": 512},
            {"index": 1, "size": 512},
            {"index": 2, "size": 512}
        ]
    },
    {
        "layers": [
            {"index": 0, "size": 1024},
            {"index": 1, "size": 1024},
            {"index": 2, "size": 1024},
            {"index": 3, "size": 1024}
        ]
    }
]

STATS = {
    0: {"variance": 1.5, "mean": 1.0},
    1: {"variance": 3.0, "mean": 1.5},
    2: {"variance": 0.5, "mean": 0.5},
    3: {"variance": 2.0, "mean": 1.2}
}

def compute_sensitivities(model_config, activation_stats):
    scores = []
    for layer in model_config["layers"]:
        idx = layer["index"]
        st = activation_stats[idx]
        scores.append(float(np.sum(st["variance"] * (st["mean"] ** 2))))
    return scores

def allocate_bits(sensitivities, bit_options, total_budget_bits):
    n = len(sensitivities)
    min_bit = min(bit_options)
    bits = [min_bit] * n
    current_bits = sum(bits)
    while current_bits < total_budget_bits:
        gains = []
        for i in range(n):
            current_b = bits[i]
            higher_options = [b for b in bit_options if b > current_b]
            if not higher_options or current_bits + (min(higher_options) - current_b) > total_budget_bits:
                gains.append(-float("inf"))
            else:
                next_b = min(higher_options)
                gains.append(sensitivities[i] * (next_b - current_b))
        if all(g == -float("inf") for g in gains):
            break
        best_i = int(np.argmax(gains))
        higher_options = [b for b in bit_options if b > bits[best_i]]
        next_b = min(higher_options)
        current_bits += (next_b - bits[best_i])
        bits[best_i] = next_b
    return bits

def emit_config_groups(model_config, assigned_bits):
    buckets = {}
    for layer, bits in zip(model_config["layers"], assigned_bits):
        buckets.setdefault(bits, []).append(layer["index"])
    groups = []
    for bits, layers in sorted(buckets.items()):
        groups.append({
            "bits": int(bits),
            "layers": sorted(layers)
        })
    return groups
