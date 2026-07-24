## Context

`log-sum-exp`, $\text{LSE}(x) = \log\sum_i e^{x_i}$, shows up everywhere a
model needs a normalizer over unnormalized log-scores (softmax denominators,
log-likelihoods, mixture models). Written literally — exponentiate every
element, sum, take the log — it breaks the moment the inputs leave a narrow
band:

- if any $x_i \gtrsim 709$, $e^{x_i}$ overflows double precision to `+inf`,
  and `log(inf) = inf` poisons the whole result even though the true answer
  is a perfectly ordinary finite number;
- if every $x_i \lesssim -745$, every $e^{x_i}$ underflows to exactly `0.0`,
  and `log(0) = -inf` — again wrong, even though the true answer is finite.

The fix doesn't change what is being computed, only *where* the huge and
tiny numbers appear. Shift every term by the maximum before exponentiating:

$$
\text{LSE}(x) = m + \log\sum_i e^{x_i - m}, \qquad m = \max_i x_i
$$

Every shifted exponent $x_i - m$ is $\le 0$, so every $e^{x_i - m}$ lands in
$(0, 1]$ — the largest term is always exactly $1$, so the sum can never
overflow and can never underflow to all-zero either. The result is
mathematically identical to the naive formula; only the floating-point path
to it is different.

## Task

Implement, in `solve.cpp`:

```cpp
double log_sum_exp(const std::vector<double>& x);
```

`x` is non-empty. Return $\log\sum_i e^{x_i}$, computed via the max-shift
trick above so it stays accurate (and finite) no matter how large or small
the elements of `x` are.

## Example

The driver (`main.cpp`, fixed) runs six hand-built fixtures:

```
normal = 3.4076059644
large_positive = 1001.1041306053
large_negative = -998.5356312159
mixed_range = 1000.0000000000
single = 42.0000000000
identical = 6.6094379124
```

`large_positive` uses $x = \{1000, 1000.5, 999\}$ — every raw $e^{x_i}$
overflows double precision to `+inf`, yet the true LSE is an ordinary number
just above $1000$ (dominated by the largest term, plus a small correction
from the other two). `large_negative` uses $x = \{-1000, -1000.5, -999\}$ —
every raw $e^{x_i}$ underflows to `0.0`, yet the true LSE is finite and close
to $-1000$. `mixed_range` uses $x = \{-1000, 500, 1000\}$: the $1000$ so
thoroughly dominates that the LSE rounds to exactly $1000$ at this
precision — the other two terms contribute less than $10^{-10}$ once
shifted.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires every printed value to be within `1e-9` of the
reference (`main.cpp` + `ref.cpp`) (`max_abs_err <= 1e-9`). The starter
returns `0.0` unconditionally, which misses every fixture; a naive
`log(sum(exp(x_i)))` implementation (no max-shift) would print `inf` on
`large_positive` and `-inf` on `large_negative` — tokens the grader's
numeric extractor doesn't even parse as numbers, so the count of printed
numbers no longer lines up with the reference's and the gate fails outright,
on top of whatever precision was already lost on the fixtures that stayed
finite.
