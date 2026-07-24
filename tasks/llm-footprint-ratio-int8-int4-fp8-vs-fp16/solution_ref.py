def compute_footprint_ratio(weights, scheme="int8", granularity="per_tensor"):
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
