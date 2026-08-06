## Context

The textbook "one-pass" variance formula computes

$$
\operatorname{Var}(x) = E[x^2] - (E[x])^2
$$

by accumulating $\sum x_i$ and $\sum x_i^2$ in a single sweep. It looks
innocent, but when the data sits on a large offset (e.g. $x_i \approx 10^8 +
\varepsilon_i$ with $\varepsilon_i$ of order 1), both $E[x^2]$ and
$(E[x])^2$ are huge numbers ($\sim 10^{16}$) whose difference is the tiny
true variance ($\sim 1$–$100$). In float64 this is catastrophic
cancellation: each term only carries about 15–16 significant decimal
digits, so the last few digits of the *result* are pure rounding noise. In
practice this formula routinely returns a **negative** "variance" on
real-world offset data, which is nonsensical since variance can never be
negative.

## Task

Implement `stable_variance(x)`:

```python
def stable_variance(x: list[float]) -> float:
    ...
```

`x` is a list of floats of floats. Return the **population variance**
(divide by $n$, i.e. `ddof=0`) as a Python `float`, computed with an
algorithm that stays accurate even when the data has a large offset.

Do **not** use the naive $E[x^2] - (E[x])^2$ formula on the raw values.
Instead, use a numerically stable formulation, e.g. shift-then-two-pass:

$$
\bar{x} = \frac{1}{n}\sum_i x_i, \qquad
\operatorname{Var}(x) = \frac{1}{n}\sum_i (x_i - \bar{x})^2,
$$

or an equivalent stable incremental scheme such as Welford's algorithm.
Because every deviation $x_i - \bar{x}$ is small once the offset has been
removed, squaring and averaging it no longer cancels significant digits.

The input array must not be modified.

## Example

```python

x = 1e8 + [1.0, -2.0, 3.0, 0.5, -1.5]
v = stable_variance(x)
# naive E[x^2] - E[x]^2 on this data is dominated by float64 rounding
# noise near 1e16 and can even come out negative; stable_variance must not.
```

## What the gate checks

The gate builds several test arrays of the form `offset + noise` with
offsets ranging from `0` up to `1e8` (and negative offsets too) and noise
of order 1–5, using a seeded generator. For each array it computes the
reference variance directly with `sum((val - sum(x) / len(x)) ** 2 for val in x) / len(x)` (itself numerically
well-behaved because Python does not use the naive cancelling formula) and
compares it against `stable_variance(x)` via the relative error

$$
\text{rel\_err} = \max_{\text{case}} \frac{|v_{\text{got}} - v_{\text{ref}}|}{|v_{\text{ref}}|}.
$$

The worst-case relative error over all test arrays must be below
$10^{-10}$. A solution that reduces to $E[x^2]-(E[x])^2$ on the raw,
large-offset data loses far more precision than that and fails the gate
(it may even return a negative number).
