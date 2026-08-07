import numpy as np


def generate_synthetic_checkpoint():
    rng = np.random.default_rng(42)
    return {
        "weight.quantized": rng.integers(-128, 127, size=(32, 32), dtype=np.int8),
        "weight.scale": rng.standard_normal((32, 1)).astype(np.float32),
        "metadata": {"format": "compressed-tensors", "version": 1}
    }


SYNTHETIC_CHECKPOINT = generate_synthetic_checkpoint()
