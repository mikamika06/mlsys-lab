## Context

An **inclusive prefix scan** turns a sequence $x_0, x_1, \dots, x_{N-1}$ into

$$y_i = \sum_{j=0}^{i} x_j , \qquad i = 0, \dots, N-1 .$$

A plain sequential running sum computes this in $N$ steps, one addition each,
but has no parallelism — step $i$ must finish before step $i+1$ starts. The
**Hillis-Steele** scan instead runs $\lceil \log_2 N \rceil$ **distance-doubling**
rounds, and every element in a round can be updated independently of the
others in that same round (this is the classic GPU-style scan). Starting from
$y^{(0)} = x$, round $k = 0, 1, 2, \dots$ produces

$$y^{(k+1)}_i = \begin{cases} y^{(k)}_i + y^{(k)}_{\,i - 2^k} & \text{if } i \ge 2^k \\ y^{(k)}_i & \text{otherwise} \end{cases}$$

After $\lceil \log_2 N \rceil$ rounds every element has accumulated
contributions from all indices $\le i$, so $y^{(\lceil \log_2 N \rceil)}$
equals the exact inclusive prefix sum — it just gets there by combining
$O(N \log N)$ pairs across $O(\log N)$ parallel rounds instead of $O(N)$
sequential steps.

## Task

Implement:

```python
def hillis_steele_scan(x: list[int]) -> list[int]:
    ...
```

- `x` — a 1-D integer list of length $N \ge 1$.
- Return a 1-D integer list `y` of the same length where `y[i]` is the inclusive prefix sum of `x[0..i]`.
- Compute it using the round-based, distance-doubling recurrence above (each round reads the *previous* round's list — do not update `y` in place while still reading old values from the same round), not a single sequential accumulator loop.


## Example

```python

x = [3, 1, 4, 1, 5, 9, 2, 6]
y = hillis_steele_scan(x)
# round 0 (shift 1): [3, 4, 5, 5, 6, 14, 11, 8]
# round 1 (shift 2): [3, 4, 8, 9, 11, 19, 17, 22]
# round 2 (shift 4): [3, 4, 8, 9, 14, 23, 25, 31]
print(y)
# [3, 4, 8, 9, 14, 23, 25, 31]
```

## What the gate checks

The grader runs several integer lists through your function — lengths that are exact powers of two, lengths that are not, `N = 1`, and lists containing negative values — and compares each output elementwise to the oracle using integer `exact_match`. Any mismatch, wrong length, wrong type (must stay integral), or an exception sets the metric to `0.0`; matching every case exactly on every fixture sets it to `1.0`. The gate requires `exact_match == 1.0`.
