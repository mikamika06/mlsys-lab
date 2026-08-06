import numpy as np

def checkpoint_stats(base_shapes, lora_shapes, dtype_bytes=2):
    """
    Calculate total size of base weights, adapter weights, and their ratio.
    """
    base_params = sum(np.prod(shape) for shape in base_shapes.values()) if base_shapes else 0
    lora_params = sum(np.prod(shape) for shape in lora_shapes.values()) if lora_shapes else 0

    base_bytes = int(base_params * dtype_bytes)
    lora_bytes = int(lora_params * dtype_bytes)
    ratio = float(lora_bytes / base_bytes) if base_bytes > 0 else 0.0

    return {
        "base_bytes": base_bytes,
        "lora_bytes": lora_bytes,
        "ratio": ratio
    }
