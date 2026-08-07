def compute_peak_memory_delta(external_bytes, in_model_bytes):
    delta = in_model_bytes - external_bytes
    ratio = in_model_bytes / (external_bytes + 1e-9)
    return {
        "external_bytes": int(external_bytes),
        "in_model_bytes": int(in_model_bytes),
        "delta_bytes": int(delta),
        "ratio": float(ratio)
    }
