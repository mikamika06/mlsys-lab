def _ref(weights, scheme, granularity):
    import numpy as np
    if scheme == "int8" and granularity == "per_tensor":
        quant_size = weights.size * np.dtype(np.int8).itemsize
        scale_bytes = np.dtype(np.float32).itemsize
    elif scheme == "int4" and granularity == "per_channel":
        n, d = weights.shape
        packed_per_channel = int((n + 1) // 2)
        quant_size = packed_per_channel * d * np.dtype(np.uint8).itemsize
        scale_bytes = d * np.dtype(np.float32).itemsize
    else:
        raise ValueError("Unsupported scheme/granularity")
    fp16_size = weights.size * np.dtype(np.float16).itemsize
    return (quant_size + scale_bytes) / fp16_size

def grade(sol, fx) -> dict:
    import numpy as np
    cases = [
        (np.random.randn(10,8), "int8", "per_tensor"),
        (np.random.randn(32,64), "int4", "per_channel"),
        (np.random.randn(7,3), "int8", "per_tensor"),
        (np.random.randn(15,5), "int4", "per_channel")
    ]
    ok = 1.0
    for weights, scheme, granularity in cases:
        try:
            got = sol.compute_footprint_ratio(weights, scheme=scheme, granularity=granularity)
        except Exception:
            ok = 0.0
            break
        ref = _ref(weights, scheme, granularity)
        if abs(got - ref) > 1e-9:
            ok = 0.0
            break
    return {"exact_match": ok}
