## Context

In capacity-provisioning and load-balancing systems the quantity that matters is
not raw throughput but **goodput** — the rate at which admitted requests actually
complete within a latency Service Level Objective (SLO).

Formally, let $n$ requests arrive at timestamps $t_1, t_2, \dots, t_n$ (sorted
ascending) with measured response latencies $\ell_1, \ell_2, \dots, \ell_n$.
A boolean admission policy selects a subset via $\text{adm}_i \in \{0,1\}$.
Given a latency SLO threshold $\tau$ and an observation window $T > 0$, the
goodput $g$ is

$$g \;=\; \frac{1}{T}\sum_{i=1}^{n}\mathbb{1}\!\bigl[\,\text{adm}_i = 1
\;\wedge\; \ell_i \le \tau\,\bigr]$$

where $\mathbb{1}[\cdot]$ is the indicator function.

A naïve scalar loop computes this in $O(n)$ interpreted operations. With
Python the entire sum vectorises into two boolean masks and a single
`sum`, eliminating the Python loop entirely.

## Task

Implement `compute_goodput`:

```python

def compute_goodput(timestamps, latencies, admitted, slo_threshold, window):
    ...
```

Return the goodput as a Python `float` — the number of admitted requests whose
latency is at most `slo_threshold`, divided by `window`. Use vectorised Python;
do not write a Python `for` loop.

## Example

```python

timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
latencies  = [0.1, 0.5, 0.3, 0.2, 0.8]
admitted   = [True, True, True, True, True]
slo        = 0.4
window     = 5.0

# Requests 0, 2, 3 have latency <= 0.4 -> 3 good
# goodput = 3 / 5.0 = 0.6
assert compute_goodput(timestamps, latencies, admitted, slo, window) == 0.6
```

## What the gate checks

The gate calls your function on **six** reference inputs (including edge cases
where all or no requests are admitted, and where no request meets the SLO) and
compares the returned `float` against a Python-computed oracle. The metric is
`exact_match`: the absolute difference must be zero (within floating-point
`1e-12` tolerance). A common mistake is forgetting the `admitted` mask and
computing throughput-of-SLO-meeting requests instead of goodput — the gate
catches this on the second test case where `admitted` filters out half the
requests.
