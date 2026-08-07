## Context

Scaled‑dot‑product attention (SDPA) is a core component of transformer models. For a query matrix $Q \in \mathbb{R}^{B\times H\times N_q\times d_k}$ and a key matrix $K \in \mathbb{R}^{B\times H\times N_k\times d_k}$ the logits are computed as

$$
\text{logits} = \frac{Q K^\top}{\sqrt{d_k}} .
$$

The denominator $\sqrt{d_k}$ is a *default scale* that stabilises gradients. Some libraries allow an explicit `scale` argument; if it is omitted the default value $1/\sqrt{d_k}$ is used.

In many applications one is interested in two numbers:

1. The actual scaling factor that was applied.
2. The largest entry of the pre‑softmax logits, i.e. $\displaystyle \max_{b,h,i,j}\text{logits}_{b,h,i,j}$.

Both values are useful for debugging and for monitoring numerical stability.

## Task

Implement `measure_sdpa_scale_and_top_logit`:

```python
def measure_sdpa_scale_and_top_logit(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], *, scale: float | None=None) -> tuple[float, float]:
    ...
```

The function receives two 4‑D list `Q` and `K`. If `scale` is `None`, the default $1/\sqrt{d_k}$ must be used; otherwise the supplied value should be applied. The function returns a tuple `(used_scale, top_logit)` where:

- `used_scale` is the floating‑point scale that was actually applied.
- `top_logit` is the maximum pre‑softmax logit over all batch, head, query and key positions.

The implementation must use only Python operations; no explicit Python loops are allowed. The result should be computed in double precision (`float64`).

## Example

```python

Q = [[[[float(0.0) for _ in range(16)] for _ in range(5)] for _ in range(4)] for _ in range(2)]
K = [[[[float(0.0) for _ in range(16)] for _ in range(7)] for _ in range(4)] for _ in range(2)]

scale, top_logit = measure_sdpa_scale_and_top_logit(Q, K)
print(scale)      # ≈ 0.25 (since 1/√16 = 0.25)
print(top_logit)  # a scalar value depending on the random data
```

## What the gate checks

Two metrics are evaluated:

- `scale_rel_err`: The relative error between the returned scale and the reference value computed by Python, must satisfy  
  $$\frac{|\,\text{returned} - \text{reference}\,|}{|\text{reference}|}\le 10^{-6}.$$

- `logit_rel_err`: The relative error of the top logit, with the same tolerance.

Both metrics are computed over several random test cases. A correct implementation will pass both gates; a broken one will fail at least one.
