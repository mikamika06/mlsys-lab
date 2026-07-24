## Context

`bfloat16` is IEEE `binary32` with the bottom 16 bits of the significand thrown
away. Both formats use **1 sign bit and 8 exponent bits**; fp32 keeps 23 stored
mantissa bits, bf16 keeps 7. So bf16 has the *same dynamic range* as fp32 —
no overflow when you cast a weight tensor down — and pays for it purely in
precision:

$$\text{bytes}(\text{bf16}) = \tfrac{1}{2}\,\text{bytes}(\text{fp32})
\qquad\Longrightarrow\qquad \text{size\_ratio} = \frac{4N}{2N} = 2 .$$

Because the top 16 bits of an fp32 word *are* the bf16 word, the conversion is
pure integer arithmetic on the bit pattern. With 7 stored mantissa bits (8
significant bits including the implicit leading 1), one ulp at magnitude
$2^{e}$ is $2^{e-7}$, so correct **round-to-nearest-even** never errs by more
than half an ulp:

$$\frac{|\hat{x}-x|}{|x|} \;\le\; 2^{-8} \;=\; 0.00390625 .$$

Naive *truncation* (just masking off the low 16 bits) doubles that to $2^{-7}$
and introduces a systematic bias toward zero — which is exactly what shows up as
drift when you cast a whole model down. The standard branch-free RNE trick is to
add a rounding bias before shifting:

$$u' = u + \bigl(\texttt{0x7FFF} + \text{lsb}\bigr), \qquad
\text{lsb} = (u \gg 16)\ \&\ 1 ,$$

then take $u' \gg 16$.

## Task

Implement the two halves of the round trip in `solve.py`.

```python
def pack_bf16(x: np.ndarray) -> np.ndarray:
    """float32 array -> uint16 array of bf16 codes, same shape."""

def unpack_bf16(codes: np.ndarray) -> np.ndarray:
    """uint16 bf16 codes -> float32 array, same shape."""
```

Requirements:

* `pack_bf16` takes a `float32` array of any shape and returns a `uint16` array
  of the **same shape** holding the bf16 bit patterns. Rounding must be
  **round-to-nearest, ties-to-even** — not truncation.
* `unpack_bf16` takes a `uint16` array and returns the exact `float32` values
  those codes denote (shift left by 16 and reinterpret).
* `unpack_bf16(pack_bf16(x))` must therefore be the correctly rounded bf16
  neighbour of every element of `x`.
* Stay vectorised; no Python loops are needed.

## Example

```python
import numpy as np

x = np.array([1.0, 1.0078125, 0.05, -3.7], dtype=np.float32)
c = pack_bf16(x)
print(c.dtype, c.nbytes, x.nbytes)     # uint16 8 16
print(unpack_bf16(c))
# [ 1.        1.0078125 0.0500488 -3.703125 ]
```

`1.0078125` is exactly representable in bf16 (it is $1 + 2^{-7}$), so it survives
the round trip untouched; `0.05` and `-3.7` are not and get nudged to their
nearest bf16 neighbours.

## What the gate checks

The grader recomputes the reference codes with an independent oracle: for every
input it forms both bf16 candidates (the truncated word and the next word up),
compares the two distances **in float64**, and picks the nearer one, breaking
exact ties toward the candidate with an even low bit. Nothing is hardcoded.

* `size_ratio` — `fixture.nbytes / packed.nbytes` on the fp32 fixture
  `W.npy`. Gate: `== 2.0` (enforced as `>= 2.0` and `<= 2.0`), so returning
  `uint32` codes or a float array fails.
* `code_exact_fraction` — fraction of packed codes equal to the oracle's codes,
  over the fixture plus several hand-built edge-case arrays (zeros, signed zeros,
  exact powers of two, exact midpoints that trigger ties-to-even, tiny and huge
  magnitudes). Gate: `>= 1.0`.
* `max_rel_err` — largest relative round-trip error over all non-zero elements.
  Gate: `<= 2^{-8} = 0.00390625`, the half-ulp bound. Truncation peaks near
  $2^{-7}$ and fails this gate on its own.

`unpack_bf16` is additionally exercised on raw codes the grader builds itself,
so a `pack` that secretly stashes fp32 data cannot pass.
