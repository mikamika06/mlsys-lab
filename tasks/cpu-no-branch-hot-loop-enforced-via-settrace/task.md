## Context

A **data-dependent branch** -- `if (x < lo) ...` -- forces the CPU's
branch predictor to guess which way execution will go before the
comparison has even finished. On a hot loop where the outcome depends on
unpredictable data, wrong guesses cost a full pipeline flush every time.
A **branchless** implementation avoids the problem entirely by computing
both possible outcomes with pure arithmetic and *selecting* between them
with a data-independent instruction (`fmin`/`fmax`/`select`) that always
takes the same number of cycles regardless of the input:

$$
\mathrm{clamp}(x, lo, hi) = \max(lo, \min(x, hi))
$$

using $\min$/$\max$ as primitives contains no branch on the *value* of
$x$ at all -- only on which of two already-computed numbers is smaller,
resolved by a single select instruction.

Because a branchy and a branchless implementation compute the exact same
output, output alone can never prove which one you wrote. This task
enforces it structurally instead, the same way the old `sys.settrace`-
based version of this exercise counted executed lines: `Guarded` hides
its float behind a `private` member, so nothing outside `sol.hpp`'s
friend functions can read it, and there is no `operator<`/`operator>`.
You cannot write `if (x < lo)` on a `Guarded` value -- it will not
compile. The only way to combine two `Guarded` values is through
`branchless_min`/`branchless_max` (real `fminf`/`fmaxf`, no branch) or
through `branchy_min`/`branchy_max` -- an escape hatch that computes the
same result with a real `if`, and increments a counter every time it's
called.

## Task

Implement:

```cpp
float clamp_branchless(Guarded x, Guarded lo, Guarded hi);
```

Return `x` clamped into `[lo, hi]` using only `branchless_min` and
`branchless_max` (both declared in `sol.hpp`, defined in `main.cpp`).
Never call `branchy_min` or `branchy_max`.

## Example

For `x = 15, lo = 0, hi = 10`: first clamp against the upper bound,
`branchless_min(x, hi) = 10`, then against the lower bound,
`branchless_max(10, lo) = 10`. Result: `10`. For `x = -3` with the same
bounds: `branchless_min(-3, 10) = -3`, then `branchless_max(-3, 0) = 0`.
Result: `0`.

## What the gate checks

`main.cpp` runs `clamp_branchless` over 8 fixed cases (inside the
range, below `lo`, above `hi`, exactly on each boundary, far outside,
and a degenerate `lo == hi == x`), prints every result, and finally
prints the total number of `branchy_min`/`branchy_max` calls made across
the whole run. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's, which reports
`branchy_calls=0`. An implementation that reaches for `branchy_min`/
`branchy_max` instead of the branchless primitives computes every clamp
value correctly but prints a nonzero `branchy_calls` count and fails --
correctness alone is not the lesson here, staying branchless is.
