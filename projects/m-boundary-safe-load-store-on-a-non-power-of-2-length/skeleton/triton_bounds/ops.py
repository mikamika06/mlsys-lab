import triton
import triton.language as tl


@triton.jit
def _safe_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    raise NotImplementedError


def safe_vector_add(x, y, n_elements, block_size=64):
    raise NotImplementedError


def catch_unmasked_store(x, n_elements, block_size=64):
    raise NotImplementedError
