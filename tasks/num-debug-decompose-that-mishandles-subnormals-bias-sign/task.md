## Context

An IEEE-754 binary32 word is a sign bit $s$, an 8-bit stored exponent $e$ and a
23-bit trailing significand field $m$. The value it denotes depends on whether
$e$ is zero:

$$
x =
\begin{cases}
(-1)^{s}\,\Big(1 + \dfrac{m}{2^{23}}\Big)\, 2^{\,e-127}, & 1 \le e \le 254 \quad\text{(normal)}\\[2mm]
(-1)^{s}\,\Big(0 + \dfrac{m}{2^{23}}\Big)\, 2^{\,1-127}, & e = 0 \quad\text{(subnormal, and }\pm 0)
\end{cases}
$$

Two details trip up almost every hand-written decoder:

1. **The bias is subtracted, not added.** The unbiased exponent is $e - 127$;
   writing $e + 127$ still "looks like a bias correction" and still produces a
   plausible integer, so nothing crashes — the reconstructed magnitudes are
   simply off by a factor of $2^{254}$ (and overflow to `inf` in float32).
2. **The leading bit is implicit only for normals.** Subnormals have no hidden
   $1$; they lead with $0$ and share the *fixed* exponent $1-127 = -126$, which
   is what lets the format degrade gracefully down to
   $2^{-149} \approx 1.4\cdot10^{-45}$ instead of falling off a cliff at
   $2^{-126}$. Hardcoding $1 + m/2^{23}$ turns every subnormal — and both signed
   zeros — into something around $2^{-126}$.

## Task

`starter.py` ships a decoder with exactly those two defects. Repair
`decompose`; `recompose` is already correct and is your specification for what
the triple must mean.

```python
def decompose(x: np.ndarray):  # -> (sign, exponent, significand)
def recompose(sign, exponent, significand) -> np.ndarray
```

* `x` is a `float32` array of **finite** values (no `inf`, no `NaN`), heavy on
  subnormals and signed zeros.
* `sign` — integer array, the raw sign bit ($0$ or $1$); $-0.0$ keeps $s = 1$.
* `exponent` — integer array of the **unbiased** exponent: $e - 127$ for
  normals, and $-126$ for every $e = 0$ word (subnormals and zeros alike).
* `significand` — `float64` array in $[0, 2)$: $1 + m/2^{23}$ for normals,
  $m/2^{23}$ for $e = 0$.

The invariant to restore is exact, not approximate:

$$
x \;=\; (-1)^{\text{sign}} \cdot \text{significand} \cdot 2^{\text{exponent}}
$$

for every element, bit for bit, including $-0.0$.

## Example

```python
import numpy as np

x = np.array([np.float32(1.5), np.float32(-0.0),
              np.uint32(0x00000001).view(np.float32)],   # smallest subnormal
             dtype=np.float32)

sign, exponent, significand = decompose(x)
# sign        -> [0, 1, 0]
# exponent    -> [0, -126, -126]
# significand -> [1.5, 0.0, 1.1920929e-07]   (= 1 / 2**23)

np.all(recompose(sign, exponent, significand).view(np.uint32) == x.view(np.uint32))
# True
```

The buggy starter reports `exponent = [254, 127, 127]` and
`significand = [1.5, 1.0, 1.0000001]`, so `recompose` returns `inf` for the
first element and a value near $2^{127}$ for the other two.

## What the gate checks

The grader builds a 4096-element fixture directly from random 32-bit patterns
(half of them with $e = 0$) and re-derives the reference triple from those same
bits with NumPy. Three gates, all exact:

* **`field_exact_fraction` == 1.0** — every `sign`, `exponent` and `significand`
  entry you return must equal the bit-level oracle exactly.
* **`recompose_byte_exact_fraction` == 1.0** — your `recompose`, fed the
  *oracle's* triple, must return a `float32` array byte-identical to the fixture
  (returning `float64` scores $0$).
* **`byte_exact_fraction` == 1.0** — the full round trip
  `recompose(*decompose(x))` must be byte-identical to `x`, so a sign flip on
  the bias cannot be cancelled by a matching flip in the packer.
