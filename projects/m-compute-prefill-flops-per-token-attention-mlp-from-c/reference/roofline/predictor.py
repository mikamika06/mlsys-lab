"""Roofline predictor for decode throughput."""

from roofline.memory import compute_decode_bytes_per_step


def predict_decode_throughput(config: dict, batch_size: int, context_len: int, peak_tflops: float, hbm_bw_gbps: float, dtype_bytes: int = 2) -> dict:
    """Predict decode performance characteristics under hardware bounds."""
    h = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    d_head = h // n_heads
    i = config.get("intermediate_size", 4 * h)

    q_flops = 2 * h * (n_heads * d_head)
    k_flops = 2 * h * (n_kv_heads * d_head)
    v_flops = 2 * h * (n_kv_heads * d_head)
    out_flops = 2 * (n_heads * d_head) * h
    attn_score_flops = 2 * n_heads * d_head * context_len
    attn_val_flops = 2 * n_heads * context_len * d_head
    attn_flops = q_flops + k_flops + v_flops + out_flops + attn_score_flops + attn_val_flops

    mlp_flops = 3 * (2 * h * i)
    flops_per_token = (attn_flops + mlp_flops) * n_layers
    total_flops = flops_per_token * batch_size

    total_bytes = compute_decode_bytes_per_step(config, batch_size, context_len, dtype_bytes)

    intensity = total_flops / total_bytes

    time_compute_sec = total_flops / (peak_tflops * 1e12)
    time_memory_sec = total_bytes / (hbm_bw_gbps * 1e9)

    step_latency_sec = max(time_compute_sec, time_memory_sec)
    bound = "compute" if time_compute_sec >= time_memory_sec else "memory"

    tokens_per_sec = batch_size / step_latency_sec

    return {
        "step_latency_sec": float(step_latency_sec),
        "tokens_per_sec": float(tokens_per_sec),
        "operational_intensity": float(intensity),
        "bound": bound
    }
