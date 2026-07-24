## Context

A CPU pipeline predicts the outcome of every branch before it is resolved, so
it can keep fetching and decoding instructions instead of stalling. An `if`
whose condition follows a fixed pattern (e.g. `i < n` in a counted loop)
predicts almost perfectly. An `if` whose condition depends on *data* the
predictor has never seen a pattern for — a per-element flag, a per-element
sign — mispredicts roughly as often as the data disagrees with the
predictor's guess, and every misprediction flushes the pipeline.

The fix is to replace the data-dependent branch with **arithmetic**: turn the
condition into a mask of all-1 or all-0 bits and combine both possible
results through it, so the CPU always executes the same straight-line code
regardless of what the data says.

$$
\text{mask} = -(\text{cond} \neq 0) \quad\text{(all 1s if true, all 0s if
false, in two's complement)}
$$
$$
\text{select}(\text{cond}, a, b) = (a \mathrel{\&} \text{mask}) \mathrel{|}
(b \mathrel{\&} \lnot\text{mask})
$$

The same idea extends to `min`/`max`: instead of comparing and branching,
extract the *sign bit* of a subtraction (which is either all-1s or all-0s
after an arithmetic right shift) and use it as the mask directly:

$$
\max(a, b) = a - \big((a - b) \mathrel{\&} ((a - b) \gg 31)\big)
\qquad
\min(a, b) = b + \big((a - b) \mathrel{\&} ((a - b) \gg 31)\big)
$$

(`>> 31` on a 32-bit signed integer is an arithmetic shift: it produces
`0xFFFFFFFF` when `a - b` is negative and `0x00000000` when it is
non-negative — exactly the mask this technique needs.)

## Task

Implement, in `solve.cpp`, using only arithmetic/bitwise mask tricks — no
`if`, `?:`, `std::min`, `std::max`, or any other construct that branches on a
*data* value (a fixed-trip-count `for (int i = 0; i < n; ++i)` loop is fine —
its trip count never depends on the array contents):

- `select_branchless(cond, a, b, out, n)` — `out[i] = cond[i] ? a[i] : b[i]`,
  built from the mask formula above. `a[i]`/`b[i]` may be negative; the mask
  trick must work on raw two's-complement bits regardless of sign.
- `clamp_branchless(x, lo, hi, out, n)` — `out[i] = clamp(x[i], lo, hi)`,
  built by composing the sign-bit `max`/`min` identities above
  (`clamp(x, lo, hi) = min(max(x, lo), hi)`).

## Example

The driver (`main.cpp`, fixed) generates 24 `(cond, a, b)` triples and 24 `x`
values from a seeded generator, and prints every input/output pair plus a
checksum:

```
sel 0 1 -812 403 -> -812
sel 1 0 217 -650 -> -650
...
sel_checksum=<n>
clamp 0 -734 -> -200
clamp 1 812 -> 300
clamp 2 55 -> 55
...
clamp_checksum=<n>
```

`cond=1` selects `a`, `cond=0` selects `b`; a `clamp` input below `lo=-200`
saturates to `-200`, above `hi=300` saturates to `300`, and one already
inside `[-200, 300]` passes through unchanged. The starter returns all
zeros, so almost every line diverges from the reference immediately.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Getting the
mask direction backwards (selecting `b` when `cond` is true), or building the
`clamp` identity with the wrong operand order in the subtraction, flips
results only on some elements — not all — which still fails the exact-match
gate on the first line it disagrees.
