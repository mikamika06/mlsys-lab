## Context

**Strength reduction** replaces an expensive operation inside a loop with a
cheaper one that produces the same values. The classic case is a loop index
multiply: on each iteration you need $idx = i \cdot s$ for a stride $s$. Because
$i$ increases by exactly $1$ every step, the product increases by exactly $s$
every step:

$$i \cdot s \;=\; (i-1)\cdot s + s.$$

So instead of recomputing the multiply, you carry an **induction variable** and
add:

```
idx = 0;
for (i = 0; i < n; i++) {
    use(idx);       // idx == i * stride
    idx += stride;  // add, not multiply
}
```

This is the transformation compilers apply automatically (`-O2` LICM +
strength reduction), but doing it by hand is only correct if it is
**result-preserving**: the sequence of $idx$ values, and everything computed
from them, must be bit-for-bit identical to the multiply form.

## Task

Implement `strided_weighted_sum` in `solve.cpp`:

```cpp
long long strided_weighted_sum(const long long* a, int n, int stride);
```

It must compute

$$acc = \sum_{i=0}^{n-1} (idx_i + 1)\cdot a[idx_i], \qquad idx_i = i \cdot stride$$

and return `acc`. Do it by strength-reducing the `i * stride` multiply into an
additive induction variable and reusing that variable for both the array
subscript and the `(idx + 1)` weight. Preconditions: `n >= 0`, `stride >= 1`,
and `(n-1)*stride` is a valid index into `a`.

The fixed driver in `main.cpp` builds a deterministic array
`a[i] = i^3 - 7i + 3` (length 64) and calls your function for several
`(n, stride)` pairs, printing each result and their total.

## Example

For `a = [3, -3, ...]` (i.e. `a[0]=3`, `a[1]=1-7+3=-3`), `n = 2`, `stride = 1`:

- `i = 0`: `idx = 0`, term `= (0+1) * a[0] = 3`
- `i = 1`: `idx = 1`, term `= (1+1) * a[1] = 2 * (-3) = -6`
- result `acc = 3 + (-6) = -3`

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and compares stdout **byte-for-byte** against
the reference build (`exact_match == 1.0`). Every printed integer — each
per-fixture sum and the total — must match exactly. The starter returns `0`, so
it prints all zeros and fails until you implement the induction-variable loop.
