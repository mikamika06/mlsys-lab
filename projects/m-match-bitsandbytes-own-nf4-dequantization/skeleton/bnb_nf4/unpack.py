import numpy as np

NF4_CODEBOOK = np.zeros(16, dtype=np.float32)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    raise NotImplementedError
