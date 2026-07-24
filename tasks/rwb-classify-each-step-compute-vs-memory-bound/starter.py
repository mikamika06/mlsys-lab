def classify_steps(steps, bytes_per_param, peak_flops, peak_bandwidth):
    """Label each serving step 'compute' or 'memory' by roofline arithmetic
    intensity.

    steps: list of (decode_tokens, prefill_tokens) pairs.
    bytes_per_param: bytes per weight element (e.g. 2.0 for fp16).
    peak_flops: accelerator peak FLOPs/second.
    peak_bandwidth: accelerator peak memory bandwidth, bytes/second.

    For each step: T = decode_tokens + prefill_tokens,
    AI(T) = 2*T / bytes_per_param, AI_ridge = peak_flops / peak_bandwidth.
    Label "compute" if AI(T) >= AI_ridge, else "memory".

    Returns a list of labels, same length and order as steps.
    """
    raise NotImplementedError('your code here')
