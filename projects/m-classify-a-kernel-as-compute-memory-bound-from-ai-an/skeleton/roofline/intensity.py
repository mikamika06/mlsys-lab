def compute_vector_add_ai(n: int, dtype_bytes: int = 4) -> float:
    """Compute AI for vector add Y = A + B."""
    raise NotImplementedError


def compute_gemv_ai(m: int, n: int, dtype_bytes: int = 4) -> float:
    """Compute AI for GEMV y = A x."""
    raise NotImplementedError


def compute_gemm_ai(m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    """Compute AI for GEMM C = A B."""
    raise NotImplementedError


def compute_bmm_ai(b: int, m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    """Compute AI for Batched GEMM C = A B."""
    raise NotImplementedError


def compute_conv2d_ai(n: int, c_in: int, c_out: int, h: int, w: int, k: int, dtype_bytes: int = 4) -> float:
    """Compute AI for 2D convolution (stride=1, padding=0, valid output)."""
    raise NotImplementedError


def compute_layernorm_ai(b: int, s: int, d: int, dtype_bytes: int = 4) -> float:
    """Compute AI for LayerNorm over hidden dimension d."""
    raise NotImplementedError


def rank_kernels_by_intensity(kernels: list) -> list:
    """Rank kernel configurations from lowest AI (most memory-bound) to highest AI (most compute-bound)."""
    raise NotImplementedError
