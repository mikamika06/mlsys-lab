## Context

NF4 ("NormalFloat4", used by `bitsandbytes` for QLoRA-style weight
quantization) is a 4-bit data type whose 16 codebook levels are **not**
evenly spaced. Neural-network weights are close to normally distributed, so
NF4 places its 16 levels at the **equal-probability-mass quantiles** of a
standard normal $N(0,1)$ — this makes the codebook information-theoretically
optimal for normally distributed inputs (each of the 16 levels covers an
equal amount of probability mass, so quantization error is spread evenly
rather than being wasted on the rare, large-magnitude tail).

**Building the 16 levels.** With `offset = 0.9677083`, and $\Phi^{-1}$ the
standard normal quantile function (inverse CDF):

$$
\begin{aligned}
v_{\text{pos}} &= \Phi^{-1}\!\big(\operatorname{linspace}(\text{offset}, 0.5,\, 9)[:-1]\big) &&\text{(8 positive values)}\\
v_{\text{neg}} &= -\Phi^{-1}\!\big(\operatorname{linspace}(\text{offset}, 0.5,\, 8)[:-1]\big) &&\text{(7 negative values)}
\end{aligned}
$$

Concatenate $v_{\text{pos}}$, the single value $0$, and $v_{\text{neg}}$ (16
values total — asymmetric: 8 non-negative, 7 strictly negative, plus exact
zero so weights near zero quantize exactly), sort ascending, then divide by
the maximum absolute value so the codebook spans $[-1, 1]$.

**Blockwise absmax quantization.** Split a weight vector $x \in
\mathbb{R}^n$ into contiguous blocks of `block_size = 64` elements (the last
block may be shorter). For block $k$ with elements $x^{(k)}$:

$$
s_k = \max_i |x^{(k)}_i| \qquad
\text{code}^{(k)}_i = \arg\min_{j \in \{0,\dots,15\}} \left| \frac{x^{(k)}_i}{s_k} - \text{level}_j \right|
$$

Each element is stored as a 4-bit code (index into the 16-level table);
$s_k$ (the per-block scale, aka "absmax") is stored separately as a
`float32`. Two 4-bit codes are packed per byte (one in the low nibble, one
in the high nibble), so $n$ elements pack into $\lceil n/2 \rceil$ bytes.

Dequantization reverses this exactly: $\hat{x}^{(k)}_i = \text{level}_{\text{code}^{(k)}_i} \cdot s_k$.

## Task

Implement three functions:

```python
def nf4_levels() -> np.ndarray:
    ...

def quantize_4bit(x: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    ...

def dequantize_4bit(packed: np.ndarray, absmax: np.ndarray, n: int, block_size: int = 64) -> np.ndarray:
    ...
```

* `nf4_levels()` — returns the 16 codebook values described above, sorted
  ascending, as a 1-D array.
* `quantize_4bit(x, block_size)` — `x` is a 1-D `float` array of length $n$.
  Returns `(packed, absmax)`:
  * `packed` — `np.uint8` array of shape `(ceil(n/2),)`. Pack the 4-bit code
    of element `2*i` into the low nibble and the code of element `2*i+1`
    into the high nibble of `packed[i]`. If `n` is odd, the last byte's high
    nibble is unused.
  * `absmax` — `np.float32` array of shape `(ceil(n/block_size),)`, one
    scale per block. Guard the degenerate case where a block is all zeros
    (use scale `1.0` so you don't divide by zero).
* `dequantize_4bit(packed, absmax, n, block_size)` — inverse of the above;
  returns a length-`n` float array approximating the original `x`.

## Example

```python
import numpy as np
x = np.array([0.1, -0.05, 0.9, -1.0, 0.0, 0.3], dtype=np.float64)

packed, absmax = quantize_4bit(x, block_size=64)
packed.dtype, packed.shape   # -> uint8, (3,)   (ceil(6/2) == 3)
absmax.shape                 # -> (1,)          (ceil(6/64) == 1)

x_hat = dequantize_4bit(packed, absmax, n=6, block_size=64)
# x_hat should be close to x -- e.g. x_hat[3] should be close to -1.0
# (the largest-magnitude element always maps to the +-1.0 codebook level)
```

## What the gate checks

* **levels_max_abs_err** — compares your `nf4_levels()` output elementwise
  against the reference derivation above (max absolute difference must be
  tiny — this is a deterministic closed-form computation, not something
  approximated by sampling).
* **rel_err** — builds several random vectors (a few thousand normally
  distributed elements each, including cases where `n` is not a multiple of
  `block_size`), round-trips them through your `quantize_4bit` +
  `dequantize_4bit`, and measures the global relative L2 error against the
  original vector, averaged over trials. It also checks that `packed` has
  dtype `uint8` and shape `(ceil(n/2),)`, and that `absmax` has shape
  `(ceil(n/block_size),)` — a wrong size counts as a failed trial.
