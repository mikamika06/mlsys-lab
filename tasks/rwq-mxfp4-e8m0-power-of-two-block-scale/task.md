## Context

The MX (microscaling) format family shares one scale across a small block of
elements (32 in the OCP MX spec), stored as an **E8M0** value — a bare 8-bit power-of-two
exponent with no mantissa. For MXFP4, each element in the block is then stored as an
E2M1 4-bit float, whose largest representable magnitude is $6.0$.

To choose the block's shared exponent, the scale must be the largest power of two that
does *not* push the block's biggest element past what E2M1 can represent. For a block
$x_1, \dots, x_{32}$ with

$$
a = \max_i |x_i|,
$$

the E8M0 exponent is

$$
e = \left\lfloor \log_2\!\left(\frac{a}{6}\right) \right\rfloor .
$$

The implied scale is $2^e$, so every element divided by $2^e$ lands within
$[-6 \cdot 2, 6 \cdot 2) $ of the representable range without overflowing — using
$\lfloor \cdot \rfloor$ (rather than rounding) guarantees the scaled block never exceeds
$6.0$ in magnitude. A block that is exactly all zero has no meaningful scale; by
convention its exponent is taken to be $0$.

## Task

Implement `mxfp4_block_exponent`:

```python
def mxfp4_block_exponent(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    ...
```

- `x`: `float64` array of shape `(n,)`, where `n` is an exact multiple of `block_size`.
- `block_size`: number of elements sharing one E8M0 exponent (always 32 in this task).

Return an integer array of shape `(n // block_size,)`: the E8M0 exponent
$e = \lfloor \log_2(a / 6) \rfloor$ for each block, where $a$ is that block's max
absolute value. If a block's max absolute value is exactly $0$, its exponent is $0$.

## Example

```python
import numpy as np

x = np.concatenate([np.full(32, 6.0), np.full(32, 12.0), np.zeros(32)])
mxfp4_block_exponent(x, 32)
# array([0, 1, 0])
# block 0: amax=6  -> log2(6/6)=log2(1)=0    -> floor -> 0
# block 1: amax=12 -> log2(12/6)=log2(2)=1   -> floor -> 1
# block 2: amax=0  -> special case           -> 0
```

## What the gate checks

The gate builds a NumPy oracle that computes, per block, `amax = max(abs(block))` and
`floor(log2(amax / 6))` (with the `amax == 0` special case), on a fixed test array that
includes an all-zero block, blocks with `amax` at clean power-of-two-of-6 boundaries, and
log-uniform random-magnitude blocks. `exact_match` — the fraction of blocks where your
integer exponent exactly equals the oracle's — must be `1.0`.
