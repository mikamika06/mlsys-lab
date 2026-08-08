import numpy as np


def generate_nf4_codebook() -> np.ndarray:
    """Generate the 16-element NF4 quantile codebook normalized to [-1, 1]."""
    raise NotImplementedError


def generate_fp4_codebook() -> np.ndarray:
    """Generate FP4 (E2M1) normalized codebook sorted in ascending order."""
    raise NotImplementedError


def generate_int4_codebook() -> np.ndarray:
    """Generate symmetric 4-bit integer codebook mapped to [-1, 1]."""
    raise NotImplementedError
