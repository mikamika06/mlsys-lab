def compute_attention_flops(B, H, N, D, causal=False):
    if not causal:
        qk_flops = 2 * B * H * N * N * D
        av_flops = 2 * B * H * N * N * D
    else:
        qk_flops = B * H * N * N * D
        av_flops = B * H * N * N * D
    return float(qk_flops + av_flops)


def derive_tflops(B, H, N, D, wall_clock_seconds, causal=False):
    total_flops = compute_attention_flops(B, H, N, D, causal=causal)
    tflops = (total_flops / wall_clock_seconds) / 1e12
    return float(tflops)
