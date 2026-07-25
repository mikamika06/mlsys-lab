## Context

`+` on floats is commutative but not associative — `(a+b)+c` and
`a+(b+c)` can round to slightly different results — which is exactly why
compilers refuse to reorder floating-point sums on their own (`-ffast-math`
aside). But the ORDER you choose determines the sum's **dependency
chain**: a left-to-right sequential sum

$$((((v_0+v_1)+v_2)+v_3)+\dots+v_{n-1})$$

has $n-1$ adds where every single one has to wait for the result of the
one before it — a critical path of length $n-1$, no matter how many
independent floating-point adders the CPU has sitting idle. Reassociating
into a **pairwise (tree) reduction** — sum adjacent pairs, then sum those
partial sums pairwise, and so on — makes each level's adds mutually
independent, so the critical path collapses to $\lceil \log_2 n \rceil$:
for $n = 64$, that's $63 \to 6$.

## Task

`sol.hpp` gives you a dependency-depth model:

- `record_add(depth_a, depth_b)` — call this once per `+` you perform,
  passing the depth of each operand (`0` for a raw, not-yet-added input
  value). Returns `1 + max(depth_a, depth_b)`.
- `critical_path_depth()` — the largest depth returned by `record_add` so
  far.
- `reset_critical_path()` — clears it.

Implement:

```cpp
float parallel_sum(const float* values, int n);
```

Sum all `n` floats (`n` is a power of two) and return the sum. Reassociate
the summation into a pairwise tree reduction — at each level, add adjacent
pairs of the current partial sums to halve the count — instead of a
sequential left-to-right sum, and report every `+` you perform through
`record_add`, mirroring your actual addition tree, so the driver can
measure how deep the true dependency chain is.

## Example

The driver (`main.cpp`, fixed) sums 64 values (`values[i] = 1.0 + 0.1 *
(i % 7)`, sum $= 82.9$) via your `parallel_sum`:

```
sum=82.900002
critical_path_depth=6
```

$64 \to 32 \to 16 \to 8 \to 4 \to 2 \to 1$ is exactly 6 halvings —
$\lceil \log_2 64 \rceil = 6$.

A left-to-right sequential sum reaches the *same* sum (up to a
sub-`1e-3` rounding difference — reassociation barely moves a
well-conditioned sum like this one) but with a critical path of `63`:

```
sum=82.899994
critical_path_depth=63
```

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires both printed numbers to satisfy `max_abs_err <=
1e-3` against the same driver linked with the reference tree reduction. A
correct sequential sum gets the `sum` value right (rounding differences
here are far below tolerance) but reports `critical_path_depth=63` instead
of `6` — a difference of `57`, which alone blows through the `1e-3`
tolerance and fails the gate. Getting the *value* right is not enough; the
addition order itself, reported through `record_add`, has to actually be a
tree.
