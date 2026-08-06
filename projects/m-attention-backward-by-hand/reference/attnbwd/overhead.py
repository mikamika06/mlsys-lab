def compute_attention_flops(B, H, N, D, pass_type="backward", recompute=True):
    if pass_type == "forward":
        return 4 * B * H * (N ** 2) * D
    elif pass_type == "backward":
        return (10 if recompute else 8) * B * H * (N ** 2) * D
    raise ValueError(f"Unknown pass_type: {pass_type}")


def recompute_overhead(B, H, N, D):
    fw = compute_attention_flops(B, H, N, D, "forward")
    bw_no = compute_attention_flops(B, H, N, D, "backward", False)
    bw_rec = compute_attention_flops(B, H, N, D, "backward", True)
    return {
        "extra_flops": bw_rec - bw_no,
        "ratio": bw_rec / bw_no,
        "recompute_vs_forward": (bw_rec - bw_no) / fw
    }
