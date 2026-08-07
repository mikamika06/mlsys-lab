import numpy as np


def apply_per_row_lora(x, adapter_ids, lora_a, lora_b, scaling):
    """Apply adapter matrices per-row based on assigned adapter IDs."""
    batch_size, in_features = x.shape
    out_features = lora_b.shape[2]
    out = np.zeros((batch_size, out_features), dtype=x.dtype)
    for i in range(batch_size):
        aid = adapter_ids[i]
        if aid < 0:
            continue
        a = lora_a[aid]
        b = lora_b[aid]
        s = scaling[aid] if isinstance(scaling, (list, tuple, np.ndarray)) else scaling
        low_rank = np.dot(x[i], a)
        out[i] = np.dot(low_rank, b) * s
    return out


def verify_batched_lora(x, adapter_ids, lora_a, lora_b, scaling, expected_out, atol=1e-5):
    """Verify that batched per-row adapter output matches expected tensor."""
    actual = apply_per_row_lora(x, adapter_ids, lora_a, lora_b, scaling)
    diff = np.abs(actual - expected_out)
    is_correct = bool(np.all(diff <= atol))
    max_err = float(np.max(diff)) if diff.size > 0 else 0.0
    return {"is_correct": is_correct, "max_error": max_err, "output": actual}
