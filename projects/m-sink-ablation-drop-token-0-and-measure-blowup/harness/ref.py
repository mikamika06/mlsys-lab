import numpy as np

np.random.seed(42)

NL = 2
NH = 4
SL = 16

DUMP = [
    {"layer": 0, "head": 0, "kept_positions": [0, 1, 2, 15]},
    {"layer": 0, "head": 1, "kept_positions": [0, 14, 15]},
    {"layer": 1, "head": 0, "kept_positions": [0, 5, 10]},
    {"layer": 1, "head": 1, "kept_positions": [0]},
]

_raw = np.random.rand(2, NH, SL, SL)
_tril = np.tril(_raw)
_sums = _tril.sum(axis=-1, keepdims=True)
_sums[_sums == 0] = 1.0
PROBS = _tril / _sums

def reconstruct_mask(num_layers, num_heads, seq_len, dump):
    mask = np.zeros((num_layers, num_heads, seq_len), dtype=bool)
    for entry in dump:
        mask[entry["layer"], entry["head"], entry["kept_positions"]] = True
    return mask

def drop_token_0(attention_probs):
    ablated = attention_probs.copy()
    ablated[..., 1:, 0] = 0.0
    sums = ablated.sum(axis=-1, keepdims=True)
    sums[sums == 0] = 1.0
    return ablated / sums

def measure_blowup(original_probs, ablated_probs):
    return float(np.max(np.abs(original_probs - ablated_probs)))
