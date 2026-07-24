## Context

Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)`
can round to different results, because each `+` rounds its result to the
nearest representable value *before* the next `+` sees it. Summing the same
multiset of numbers in a different order therefore changes the answer — and
some orders are dramatically worse than others.

The failure mode is starkest when a running total grows much larger than the
values still being added to it. A `float` (single precision) has 23
explicit mantissa bits, so its spacing between representable values (the
ULP) at magnitude $2^{25}$ is $2^{25-23} = 4$. Adding $1.0$ to a value that
is already an exact multiple of 4 near that magnitude rounds to the
*nearest* representable float — which is the unchanged original value,
because $1.0$ is closer to $0$ than to $4$. The addition is not wrong, not
undefined behavior, not a bug in the hardware: it is IEEE-754 round-to-
-nearest doing exactly what it's specified to do, and it silently discards
the entire contribution.

## Task

Implement, in `solve.cpp`:

```cpp
std::vector<double> sum_ordering_rel_errors(const std::vector<float>& x);
```

Sum `x` four different ways, with the running accumulator declared as
`float` in every case (never `double` — that would hide the effect this
task is about):

0. **forward** — left to right.
1. **reverse** — right to left.
2. **pairwise** — recursively split `[lo, hi)` at the midpoint, sum each
   half, add the two half-sums. (Depth $O(\log n)$ instead of $O(n)$, so
   the running accumulator at any one addition never grows as large
   relative to the next term.)
3. **kahan** — Kahan compensated summation: track a running compensation
   term `c` that captures the low-order bits an ordinary `float` addition
   would drop, and feed them back into the next term before adding.

Separately, compute a high-precision reference by summing the same values
left to right with a `double` accumulator. Return the four **relative**
errors, `|sum32 - ref64| / |ref64|`, in the order
`[forward, reverse, pairwise, kahan]`.

## Example

The driver (`main.cpp`, fixed) builds `x = [2^25, 1.0, 1.0, ..., 1.0]`
(100000 ones after the leading $2^{25} = 33554432$). The true sum,
$33654432$, is exactly representable in both `float` and `double`:

```
n=100001
forward rel_err=2.9713768457e-03
reverse rel_err=0.0000000000e+00
pairwise rel_err=1.1885507383e-07
kahan rel_err=0.0000000000e+00
```

`forward` starts the accumulator at $2^{25}$ and adds `1.0` a hundred
thousand times — every single one of them rounds away as described above,
so the accumulator never leaves $2^{25}$ and the relative error is the
entire missing $100000 / 33654432$. `reverse` adds the hundred thousand
`1.0`s together first (reaching exactly $100000.0$, well inside `float`'s
exact-integer range) and only adds the huge value once, at the very end —
zero error. `pairwise` never lets the accumulator grow much larger than the
term it's adding, so its error is four orders of magnitude smaller than
forward's, but the merges near the root of the recursion still aren't
perfectly free of rounding. `kahan`'s compensation term recovers the exact
answer even summing strictly left to right.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires every printed relative error to be within `1e-9` of
the reference (`main.cpp` + `ref.cpp`) (`max_abs_err <= 1e-9`). Accumulating
any of the four sums in `double` instead of `float` erases the very effect
being measured and prints near-zero error everywhere, disagreeing with the
reference's `forward` and `pairwise` values; swapping the pairwise split
point or the Kahan compensation sign produces a different (and visibly
wrong) error in those two slots specifically.
