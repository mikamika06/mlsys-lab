## Context

Libraries like `torchao` and `bitsandbytes` shrink AdamW's memory footprint
by storing the optimizer's momentum $m$ and second moment $v$ in 4-bit
blockwise-quantized form instead of fp32. On every step, the stored state is
**dequantized** back to float, the usual AdamW math is done in float, and the
*updated* $m,v$ are **requantized** back to 4-bit for storage. This task asks
you to implement exactly that round trip for a single optimizer step.

### 4-bit blockwise symmetric quantization (used for both $m$ and $v$)

Split a 1-D array into contiguous blocks of `block_size` elements. For block
$b$:
$$
s_b = \frac{\max_i |x_i^{(b)}|}{7}\quad(\text{use } s_b=1 \text{ if the block is all-zero})
$$
$$
c_i^{(b)} = \mathrm{clip}\big(\mathrm{round}(x_i^{(b)} / s_b),\, -7,\, 7\big)
$$
Pack two consecutive codes into one byte: offset each code by `+8` (landing
in `[1, 15]`), element `2k` goes in the low nibble, element `2k+1` in the
high nibble: `byte = low | (high << 4)`. Assume length and `block_size` are
both even. Dequantization unpacks the nibbles, subtracts 8, and multiplies
by the block's scale: $\hat{x}_i^{(b)} = c_i^{(b)} \cdot s_b$.

### One AdamW step (decoupled weight decay), given dequantized $m_{\text{prev}}, v_{\text{prev}}$

$$
m = \beta_1 m_{\text{prev}} + (1-\beta_1) g \qquad v = \beta_2 v_{\text{prev}} + (1-\beta_2) g^2
$$
$$
\hat{m} = \frac{m}{1-\beta_1^{t}} \qquad \hat{v} = \frac{v}{1-\beta_2^{t}}
$$
$$
p_{\text{new}} = p \cdot (1 - \text{lr} \cdot \lambda) - \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$
where $t$ is the (1-indexed) step count and $\lambda$ is `weight_decay`.

## Task

Implement:

```python
def adamw_4bit_step(
    p: np.ndarray, grad: np.ndarray,
    m_packed: np.ndarray, m_scales: np.ndarray,
    v_packed: np.ndarray, v_scales: np.ndarray,
    step: int, block_size: int = 32,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.01,
) -> dict:
    ...
```

* `p`, `grad` — 1-D `float64` arrays, `len(p)` divisible by `block_size`.
* `m_packed`, `v_packed` — `uint8` arrays of length `len(p)/2`: the previous
  momentum/second-moment, 4-bit blockwise-quantized and nibble-packed as
  specified above.
* `m_scales`, `v_scales` — `float32` arrays of length `len(p)/block_size`:
  the per-block scales for `m_packed`/`v_packed`.
* `step` — the 1-indexed step count $t$ for bias correction.

Return a `dict`:
1. Dequantize `m_packed`/`m_scales` and `v_packed`/`v_scales` to get
   $m_{\text{prev}}, v_{\text{prev}}$.
2. Apply the AdamW step above to get $m, v, p_{\text{new}}$.
3. Requantize $m$ and $v$ with the same 4-bit blockwise scheme.
4. Return `{"p_new": p_new, "m_packed": ..., "m_scales": ..., "v_packed": ..., "v_scales": ...}`.

Vectorised NumPy only; no explicit Python loops over elements.

## What the gate checks

The grader builds a deterministic parameter vector and a short history of
prior gradients (fixed seed) to obtain a realistic pre-quantized $m,v$ state,
then calls your function for one more step and compares against an
independent NumPy oracle implementing the exact scheme above:

* **param_rel_err** — relative L2 error between your `p_new` and the
  oracle's `p_new`.
* **state_rel_err** — relative L2 error between the *dequantized* $[m, v]$
  you produced (by unpacking your returned `m_packed`/`m_scales` and
  `v_packed`/`v_scales`) and the oracle's dequantized $[m, v]$ after the same
  step.

Both must stay within a tight tolerance — the algorithm is fully
deterministic, so any correct implementation lands within rounding-level
distance of the oracle.
