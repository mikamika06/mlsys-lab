import numpy as np


def compute_decode_bandwidth_floor(
    batch_size,
    current_seqlens,
    num_kv_heads,
    head_dim,
    num_layers,
    dtype_bytes,
    memory_clock_ghz,
    bus_width_bits,
    achieved_time_ms,
):
    seqlens = np.asarray(current_seqlens, dtype=np.int64)
    if len(seqlens) != batch_size:
        raise ValueError("Length of current_seqlens must match batch_size")

    bytes_per_element = dtype_bytes
    element_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_element
    total_kv_bytes_read = int(np.sum(seqlens)) * num_layers * element_bytes_per_token_per_layer

    bus_width_bytes = bus_width_bits / 8.0
    theoretical_bw_gbps = memory_clock_ghz * 2.0 * bus_width_bytes

    achieved_time_sec = achieved_time_ms / 1000.0
    achieved_bw_gbps = (total_kv_bytes_read / 1e9) / achieved_time_sec if achieved_time_sec > 0 else 0.0

    utilization_floor = achieved_bw_gbps / theoretical_bw_gbps if theoretical_bw_gbps > 0 else 0.0

    return {
        "total_kv_bytes_read": int(total_kv_bytes_read),
        "theoretical_bw_gbps": float(theoretical_bw_gbps),
        "achieved_bw_gbps": float(achieved_bw_gbps),
        "utilization_floor": float(utilization_floor),
    }
