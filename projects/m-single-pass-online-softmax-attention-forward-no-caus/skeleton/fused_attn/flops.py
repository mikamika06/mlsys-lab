def compute_attention_flops(B, H, N, D, causal=False):
    """
    Returns theoretical FLOPs for attention forward pass.
    """
    raise NotImplementedError


def derive_tflops(B, H, N, D, wall_clock_seconds, causal=False):
    """
    Returns TFLOPS achieved given execution time in seconds.
    """
    raise NotImplementedError
