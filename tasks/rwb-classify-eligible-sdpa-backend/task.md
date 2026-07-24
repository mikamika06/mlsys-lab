## Context

In PyTorch's scaled dot‑product attention (SDPA) implementation, the backend that actually computes the attention matrix is chosen automatically based on several properties of the input tensors and the execution device. The three possible backends are

* **flash** – a highly optimised CUDA kernel that requires 16‑bit precision and no arbitrary mask.
* **mem_efficient** – a memory‑efficient CUDA implementation that also needs 16‑bit precision but can handle larger head dimensions than flash.
* **math** – the generic CPU/GPU implementation that works for any dtype, mask or head size.

The selection rules used by PyTorch are:

1. If the tensors reside on a CPU (`device_is_cpu == True`) the math backend is always chosen.
2. On GPU, an arbitrary attention mask (`has_attn_mask == True`) forces the fallback to math.
3. When no mask is present and the dtype is either `float16` or `bfloat16`, the head dimension determines the backend:
   * $\\text{head_dim} \\le 128$ → **flash**
   * $128 < \\text{head_dim} \\le 256$ → **mem_efficient**
   * otherwise → **math**

The flag `is_causal` is ignored by the current selection logic but is part of the public API.

## Task

Implement the function

```python
def classify_backend(dtype: str,
                     head_dim: int,
                     has_attn_mask: bool,
                     is_causal: bool,
                     device_is_cpu: bool) -> str:
    ...
```

It should return one of the strings `"flash"`, `"mem_efficient"` or `"math"` according to the rules above. The function must be pure and have no side effects.

## Example

```python
print(classify_backend("float16", 64, False, True, False))
# flash

print(classify_backend("float32", 128, False, False, False))
# math

print(classify_backend("bfloat16", 200, False, False, False))
# mem_efficient
```

## What the gate checks

The grader computes the expected backend for each test case using the exact rule table described in the context. Your implementation must match that output exactly; any deviation causes a failure.
