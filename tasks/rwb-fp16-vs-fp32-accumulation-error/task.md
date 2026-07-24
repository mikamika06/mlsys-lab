## Context

Attention kernels commonly store activations in a low-precision format
(`float16`) to save memory and bandwidth, but the choice of **accumulation**
precision — the dtype of the running sums inside the dot products, softmax
reduction, and weighted-value sum — is a separate decision with a real
numerical cost. Rounding the accumulator itself to `float16` after every
partial update compounds rounding error across the reduction; keeping the
accumulator in `float32` (even while inputs stay in `float16`) keeps that
error close to negligible. This is exactly why production kernels
(FlashAttention included) upcast to `float32` for accumulation even when
storage and matmul inputs are `float16`.

For a dtype $\tau$ (either `float16` or `float32`), define single-precision
attention with **every** running sum rounded to $\tau$ at each step. The
score between query $i$ and key $j$ accumulates as

$$
s_0 = 0, \qquad s_{k+1} = \operatorname{round}_\tau\!\big(s_k + \tau(Q_{i,k})\,\tau(K_{j,k})\big), \qquad S_{ij} = \operatorname{round}_\tau\!\Big(\frac{s_d}{\sqrt{d}}\Big),
$$

over $k = 0, \dots, d-1$ (left to right). Softmax is computed row-wise with
the same per-step rounding, using a max-shift for stability:

$$
m_i = \max_j S_{ij}, \qquad
e_{ij} = \operatorname{round}_\tau\!\big(\exp(S_{ij} - m_i)\big), \qquad
l_i = \operatorname{round}_\tau\text{-accumulated sum of } e_{ij} \text{ over } j,
$$

$$
P_{ij} = \operatorname{round}_\tau\!\Big(\frac{e_{ij}}{l_i}\Big).
$$

Finally the output accumulates the same way over keys:

$$
o_0 = 0, \qquad o_{k+1} = \operatorname{round}_\tau\!\big(o_k + \tau(P_{i,j_k})\,\tau(V_{j_k, \cdot})\big), \qquad O_i = o_m .
$$

The **relative error** of a precision path against the exact `float64`
computation of the same attention is

$$
\mathrm{rel\_err}(\tau) = \frac{\lVert O^{(\tau)} - O^{(64)} \rVert_2}{\lVert O^{(64)} \rVert_2}.
$$

## Task

Implement `fp16_vs_fp32_attention_error(Q, K, V)`:

```python
def fp16_vs_fp32_attention_error(Q, K, V):
    ...
```

Inputs are NumPy arrays `Q` $(n,d)$, `K` $(m,d)$, `V` $(m,d_v)$.

1. Compute the exact reference output $O^{(64)}$ by running attention with
   every input and every intermediate value in `float64`.
2. Compute $O^{(16)}$: attention run with $\tau = $ `float16` — cast `Q`,
   `K`, `V` to `float16`, and accumulate the score dot product, the
   softmax normalizer sum, and the output dot product **step by step**,
   rounding the running sum to `float16` after every single addition
   (do **not** use `np.dot`/`@`/`np.sum` for these three reductions —
   those routines may silently accumulate in a wider dtype internally,
   which hides exactly the error this task measures).
3. Compute $O^{(32)}$ the same way with $\tau = $ `float32`.
4. Return `(fp16_rel_err, fp32_rel_err)`: the relative L2 error (as
   defined above) of $O^{(16)}$ and $O^{(32)}$ against $O^{(64)}$,
   respectively, as Python floats.

## Example

```python
import numpy as np

Q = np.load("fixtures/q.npy")
K = np.load("fixtures/k.npy")
V = np.load("fixtures/v.npy")

fp16_err, fp32_err = fp16_vs_fp32_attention_error(Q, K, V)
# fp16_err is noticeably larger than fp32_err -- rounding the accumulator
# itself to float16 at every step compounds error across the reduction,
# while a float32 accumulator stays close to the float64 truth.
```

## What the gate checks

The gate loads the committed `q.npy`/`k.npy`/`v.npy` fixture and computes
its own `float64`, step-by-step-rounded `float16`, and step-by-step-rounded
`float32` attention outputs independently, exactly as specified above. It
compares your two reported relative errors against these independently
computed values. The gate fails (reported as an infinite error) if either
reported value doesn't match the oracle's own value closely, if
`fp32_rel_err` is not strictly smaller than `fp16_rel_err`, or if
`fp32_rel_err >= 1e-3`. Implementations that use `@`/`np.dot`/`np.sum` for
the `float16` reductions (which numpy silently computes with a wider
internal accumulator) will report an artificially small `fp16_rel_err`
that does not match the oracle's genuinely step-rounded value, and will
fail the gate.
