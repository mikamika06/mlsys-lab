import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _silu_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-x))
        y = x * s
        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.jit
    def _silu_backward_kernel(x_ptr, grad_out_ptr, grad_x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        g_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-x))
        g_x = g_out * s * (1.0 + x * (1.0 - s))
        tl.store(grad_x_ptr + offsets, g_x, mask=mask)


def triton_silu_forward(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    if HAS_TRITON and x.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _silu_forward_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        for pid in range(num_blocks):
            b_start = pid * BLOCK_SIZE
            b_end = min(b_start + BLOCK_SIZE, n_elements)
            xb = x_flat[b_start:b_end]
            sb = torch.sigmoid(xb)
            y_flat[b_start:b_end] = xb * sb
    return y


def triton_silu_backward(x: torch.Tensor, grad_output: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    grad_x = torch.empty_like(x)
    n_elements = x.numel()
    if HAS_TRITON and x.is_cuda:
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _silu_backward_kernel[grid](x, grad_output, grad_x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        x_flat = x.view(-1)
        go_flat = grad_output.view(-1)
        gx_flat = grad_x.view(-1)
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        for pid in range(num_blocks):
            b_start = pid * BLOCK_SIZE
            b_end = min(b_start + BLOCK_SIZE, n_elements)
            xb = x_flat[b_start:b_end]
            gob = go_flat[b_start:b_end]
            sb = torch.sigmoid(xb)
            gx_flat[b_start:b_end] = gob * sb * (1.0 + xb * (1.0 - sb))
    return grad_x
