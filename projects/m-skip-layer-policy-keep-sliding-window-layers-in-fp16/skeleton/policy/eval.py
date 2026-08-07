import numpy as np

def simulate_kv_cache_output(layers, dtypes, seq_len, hidden_dim, seed=42):
    """
    Simulate a simplified forward pass over layers and return final hidden states.
    For each layer:
    - Compute K, V with random normal weights (scaled by 1/sqrt(hidden_dim)).
    - Add noise to K and V depending on dtype:
      - "float8": 5% uniform proportional noise (e.g., k += noise * k * 0.05)
      - "float16": 0.1% uniform proportional noise
    - Compute causal attention over V (apply sliding window mask if "kind" == "sliding").
    - Return accumulation of outputs.
    """
    raise NotImplementedError

def eval_rel_err(layers, dtypes, seq_len, hidden_dim):
    """
    Compare the mixed-precision output against an ideal "float32" reference output.
    Return relative difference: ||test - ideal|| / (||ideal|| + 1e-9).
    """
    raise NotImplementedError
