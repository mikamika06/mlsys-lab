import numpy as np


def undo_rope_permutation(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Undo GGUF RoPE permutation on tensor w to restore HuggingFace layout."""
    raise NotImplementedError
