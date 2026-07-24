## Context

Rounding once is not the same as rounding twice.

Let $x$ be an exact real number. There are two ways to land it in binary32:

* **Direct** — round $x$ to the nearest binary32 value, ties to even.
* **Double** — round $x$ to the nearest binary64 value first (ties to even), then round
  *that* to binary32 (ties to even).

Hardware and compilers do this by accident all the time: an intermediate result is
kept in a wider register (x87's 80-bit stack, an FMA accumulator, a float64 temporary
in Python) and only later narrowed to the storage format. The folklore claim "extra
precision can only help" is false — the two paths can disagree.

The mechanism is the **tie**. Round-to-nearest-even only breaks a tie specially when
the value sits *exactly* halfway between two representables. The first rounding can
manufacture such a tie out of a value that was not one, and then the second rounding
resolves it in the wrong direction.

Concretely, on $[1,2)$ the binary32 grid has spacing $2^{-23}$ and the binary64 grid
has spacing $2^{-52}$. A number just barely above the binary32 halfway point
$1 + 2^{-24}$ must round *up* to $1 + 2^{-23}$. But if it is close enough to that
halfway point, rounding to binary64 first snaps it exactly *onto* $1 + 2^{-24}$ —
which is now a perfect tie for binary32, and ties-to-even sends it *down* to $1.0$.

$$
x \;=\; 1 + 2^{-24} + \varepsilon,\qquad 0 < \varepsilon < 2^{-53}
$$

## Task

Implement `double_rounding_counterexample()`:

```python
def double_rounding_counterexample() -> tuple[int, int, float, float]:
    ...
```

It takes no arguments and returns a 4-tuple `(num, den, direct, doubled)`:

* `num`, `den` — Python integers describing the exact rational
  $x = \texttt{num}/\texttt{den}$. `den` must be a power of two and $x$ must satisfy
  $1 \le x < 2$ (so that both formats stay normal). Arbitrary precision is fine —
  these are exact integers, not floats.
* `direct` — the binary32 value obtained by rounding $x$ **directly**, ties to even.
* `doubled` — the binary32 value obtained by rounding $x$ to binary64 first and then
  narrowing that binary64 to binary32, each step ties to even.

Your $x$ must be a genuine counterexample: `direct` and `doubled` must be **different**
binary32 numbers.

## Example

The shape of an answer (numbers below are illustrative, not the graded ones):

```python
num, den, direct, doubled = double_rounding_counterexample()
print(Fraction(num, den))       # 1 + 2^-24 + something tiny
print(np.float32(direct).view(np.uint32) != np.float32(doubled).view(np.uint32))
# True  -- the two rounding paths disagree
```

Useful facts: $1 + 2^{-24}$ is exactly representable in binary64 (it needs 25
significand bits, and binary64 has 53), but in binary32 it is exactly the midpoint
between $1.0$ and $1 + 2^{-23}$. The significand of $1.0$ is even; the significand of
$1 + 2^{-23}$ is odd.

## What the gate checks

* `valid_input` — `den` is a power of two, `num` and `den` are integers, and
  $1 \le \texttt{num}/\texttt{den} < 2$.
* `paths_differ` — the grader recomputes both roundings itself. The direct binary32
  rounding is done with **exact integer arithmetic** on the rational (no float ever
  touches it); the binary64 step is done the same way and the final narrowing is done
  by **NumPy's real `np.float32()` cast**. The gate passes only if the two resulting
  bit patterns actually differ, i.e. your $x$ really is a counterexample.
* `direct_match` / `double_match` — your reported `direct` and `doubled`, viewed as
  raw `uint32` bit patterns, must equal the grader's independently computed ones.

Nothing is hardcoded: the grader derives everything from the rational you hand it.
