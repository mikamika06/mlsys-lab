## Context

In modern transformer implementations, the scaled dot‑product attention (SDPA) kernel can be executed by several backends depending on the shape of the query/key/value tensors and the mask type. The library exposes a dispatch function that selects the most efficient backend given four constraints:

- `head_dim`: the dimensionality of each head,
- `dtype`: the NumPy dtype of the attention matrices (`float16`, `bfloat16`, `float32`),
- `mask_type`: either `"causal"` or `"full"`,
- `causal`: a boolean flag that indicates whether the causal mask should be applied.

The dispatch rules are:

1. **Flash** – used when the head dimension is a multiple of 64, the dtype is either `float16` or `bfloat16`, and the mask is causal.
2. **Mem‑efficient** – used for small heads (≤ 256) with `float32` data and a full mask.
3. **Math** – fallback for all other configurations.

These rules are deterministic and must be implemented exactly; any deviation will lead to sub‑optimal performance in production.

## Task

Implement the function `pick_backend(head_dim: int, dtype: str, mask_type: str, causal: bool) -> str` that returns one of `"flash"`, `"mem_efficient"` or `"math"` according to the rules above. The function must be pure and have no side effects.

## Example

```python
>>> pick_backend(128, "float16", "causal", True)
'flash'
>>> pick_backend(64,  "float32", "full",   False)
'mem_efficient'
>>> pick_backend(512, "int8",    "causal", True)
'math'
```

## What the gate checks

The grader evaluates a set of representative configurations and compares your output with an oracle that implements the same eligibility rules. The comparison is exact: every returned string must match the expected backend. No tolerance or approximate matching is allowed.
