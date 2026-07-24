## Context

Plain sequential floating-point summation accumulates rounding error: each
addition $s \leftarrow s + x_i$ rounds to the nearest representable
float, silently discarding the low-order bits of the true mathematical
sum. When one term is enormous and the rest are tiny, this is
catastrophic: adding $1.0$ to $10^{16}$ in `float64` returns exactly
$10^{16}$ again — the $1.0$ is entirely below the representable ULP and
vanishes without a trace.

**Kahan summation** fixes this by tracking, at each step, the low-order
bits that got rounded away, and folding them back in on the next
addition:

$$
\begin{aligned}
y &= x_i - c \\
t &= s + y \\
c &= (t - s) - y \\
s &= t
\end{aligned}
$$

Here $c$ is the running compensation: $(t - s)$ recovers (in
floating-point) what was actually added to $s$, so $(t-s) - y$ is exactly
the part of $y$ that got rounded away. That lost part is subtracted from
next input on the next iteration, so it isn't lost after all.

## Task

Implement `kahan_sum`:

```python
def kahan_sum(x: np.ndarray) -> float:
    ...
```

* `x` — a 1-D `float64` NumPy array.
* Returns the sum of `x` as a Python `float`.

You must implement this as a **real, explicit per-element Python `for`
loop** carrying out the four-line update above — not `np.sum`, not
`math.fsum`, not `np.cumsum`. The whole point of the exercise is the
compensation logic itself.

## Example

```python
import numpy as np
x = np.concatenate([[1e16], np.ones(2000), [-1e16]])
print(sum(x))         # -> 0.0            (every +1.0 got lost)
print(kahan_sum(x))   # -> 2000.0         (exactly recovered)
```

## What the gate checks

Two independent gates, both computed against real oracles:

* **line_count** — the grader wraps your call to `kahan_sum` with
  `sys.settrace` and counts Python-level line events. A real per-element
  loop over a few-thousand-element array emits many thousands of line
  events; a vectorised `np.sum`/`math.fsum` call emits almost none (the
  work happens in C). Must be `>= 1000`.
* **rel_err** — on three deterministic catastrophic-cancellation
  fixtures (a huge value, many `+1.0` terms, then the huge value
  negated, at three different sizes/magnitudes), the worst-case relative
  error of your result against `math.fsum` — Python's real,
  correctly-rounded summation function, used purely as the accuracy
  oracle here, not as an implementation shortcut. Must be below `1e-10`.
