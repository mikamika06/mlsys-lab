## Context

Summing an array into a single running total, `s += x[i]`, is a
**left-associated** reduction: every addition depends on the result of
the previous one, so the critical path — the longest chain of
data-dependent operations — has length $n - 1$ for $n$ elements, no
matter how many execution ports the CPU has.

**Reassociating** the same sum into a balanced binary tree —
$(x_0+x_1) + (x_2+x_3) + \dots$, combined level by level — changes
nothing about *which* numbers get added (floating-point order can matter
in general, but this task uses exact integer-valued doubles so the total
is identical either way), only the *shape* of the dependency graph. Each
level of the tree can run independently of the others at that same
level, so the critical path collapses from $O(n)$ to the height of the
tree:

$$
\text{depth} = \lceil \log_2 n \rceil .
$$

This task makes that critical-path length *real and measurable*: every
value is a `Tracked{value, depth}` pair (declared in `sol.hpp`), and its
`operator+` computes `depth = 1 + max(a.depth, b.depth)` — exactly the
standard "longest path in a DAG of additions" recurrence. A serial
accumulator's depth grows by 1 on every add, ending at $n-1$; a balanced
tree ends at $\lceil \log_2 n \rceil$.

## Task

Implement

```cpp
Tracked reduce_balanced_tree(const Tracked* x, int n);
```

by **reassociating** the reduction into a balanced binary tree: split
`x[0..n)` into two halves (`x[0..mid)` and `x[mid..n)` with
`mid = n / 2`), reduce each half **recursively** with
`reduce_balanced_tree`, then combine the two partial results with one
final `operator+`. The base case is `n == 1`: return `x[0]` unchanged.
The returned `.value` must be the exact sum of `x[0..n)`; the returned
`.depth` is whatever falls out of actually building the tree — you don't
set it directly.

## Example

For `x = {1,2,3,4}` (depth 0 each): `left = reduce({1,2})` has depth 1
(`1+2`), `right = reduce({3,4})` has depth 1 (`3+4`), and
`left + right` has depth `1 + max(1,1) = 2` — matching
$\lceil \log_2 4 \rceil = 2$, versus depth 3 for summing all four
serially.

## What the gate checks

`main.cpp` builds two deterministic integer-valued fixtures (so the sums
are exact regardless of association order): $N_1 = 4096$, a power of two
where a correctly balanced tree has **exact** depth
$\log_2 4096 = 12$, and $N_2 = 1000$, a non-power-of-two where the exact
depth is $\lceil \log_2 1000 \rceil = 10$. It prints both sums and both
resulting depths. A solution that ignores the tree structure and sums
everything into one running `Tracked` gets the **values** exactly right
(`-2000.000000` and `-869.000000`) but depths of `4096` and `1000`
instead of `12` and `10` — the point of this task is the shape of the
dependency graph, not just the final number, and the printed depth
mismatch fails the gate even though the sums match. The grader compiles
your `.cpp` with the real local `clang++`, runs it, and requires the
printed output to match the reference's exactly.
