import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    raise NotImplementedError
