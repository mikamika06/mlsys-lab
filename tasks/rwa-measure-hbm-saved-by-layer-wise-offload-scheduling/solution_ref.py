def measure_hbm_saved(n_layers, active_layers, per_layer_kv_bytes):
    full_bytes = n_layers * per_layer_kv_bytes
    resident_bytes = active_layers * per_layer_kv_bytes
    ratio = resident_bytes / full_bytes if full_bytes else 0.0

    return {
        "peak_resident_bytes": resident_bytes,
        "resident_ratio": ratio,
    }
