## Context

A common compiler optimization ("kernel fusion", as done by XLA, TVM and
similar tensor compilers) merges an elementwise multiply with the
reduction that follows it, so the intermediate product array never has to
be materialised in memory:

$$
y = \sum_{i=1}^{N} x_i w_i .
$$

Instead of computing the product array and calling a separate,
numerically-careful `sum` routine, the fused kernel accumulates the
running total *as it produces each product*, inside a single loop over
memory order, using one accumulator register of the kernel's native width
(here, `float32`).

That accumulator-based fusion silently changes the **order** in which
terms are summed compared to an unfused implementation (which would call
a library `sum` that internally uses a numerically safer order, such as
pairwise/tree summation). Floating-point addition is not associative, so
this reordering is not just a stylistic difference — it changes the
result.

The failure mode is worst when the data has a wide dynamic range: once
the running accumulator reaches some large magnitude $A$, IEEE 754
round-to-nearest means

$$
A + \delta = A \quad \text{whenever} \quad |\delta| < \tfrac{1}{2}\,\mathrm{ulp}(A).
$$

So if a huge term arrives early and is followed by many terms individually
smaller than half its ULP, a sequential accumulator silently drops
**every one of them** — not because of accumulated rounding, but because
each individual addition is an exact no-op.

## Task

Implement `fused_dot_reduce`:

```python
def fused_dot_reduce(x: list[float], w: list[float]) -> float:
    ...
```

* `x`, `w` — 1-D array-likes of the same length, treated as `float32`.
* Returns `sum(x * w)` as a plain Python `float`, computed with `float32`
  arithmetic throughout (matching the fused kernel's native accumulator
  width) but using a **numerically safe reduction order**: combine
  partial sums pairwise (tree reduction) rather than folding every term
  into one running sequential accumulator.

A correct pairwise reduction never lets a single accumulator absorb more
than $O(\log_2 N)$ additions before being combined with another
comparably-scaled partial sum, so rounding error grows as $O(\log N)$
instead of $O(N)$.

## Example

```python

huge = float(2e8)                  # one huge-magnitude term
smalls = [1.0] * 10_000   # many far smaller terms
x = [huge] + smalls
w = [1.0] * len(x)

y = fused_dot_reduce(x, w)
print(y)              # close to 2e8 + 10_000 = 200_010_000

# A naive sequential float32 accumulator instead returns ~2e8 exactly:
# every `+1.0` is individually smaller than half of ulp(2e8) and is
# rounded away as a complete no-op, so all 10_000 small terms vanish.
```

## What the gate checks

The grader builds several adversarial cases: one large-magnitude term
followed by many terms individually smaller than half that term's
`float32` ULP, plus a couple of plain random cases for general sanity. For
each case it computes the true value as the exact `float64` sum of the
given `float32` inputs (`sum` on the values cast to `float64`) — a real
oracle, never your algorithm.

Your output is compared against that oracle with the scorer `rel_err`
(relative L2/absolute error). The gate requires

$$
\mathrm{rel\_err} \le 10^{-6}.
$$

A sequential single-accumulator reduction fails this by several orders of
magnitude on the adversarial cases (it can drop nearly all of the small
terms' contribution); a pairwise/tree reduction passes comfortably.
