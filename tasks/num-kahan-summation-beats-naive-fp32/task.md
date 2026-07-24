## Context

Summing a list of `float32` numbers one at a time in the obvious way

$$
s_0 = 0, \qquad s_k = s_{k-1} + x_k
$$

loses information whenever $|s_{k-1}|$ is much larger than $|x_k|$: `float32`
only has about 7 decimal digits of precision, so once the running sum's
magnitude grows large, adding a small $x_k$ can round back to $s_{k-1}$
exactly, silently discarding $x_k$.

**Kahan (Neumaier) compensated summation** fights this by tracking a running
correction term $c$ that captures the low-order bits lost at each step, and
folding it back in:

$$
\begin{aligned}
t &= s + x_k \\
c &\mathrel{+}= \begin{cases} (s - t) + x_k & |s| \ge |x_k| \\ (x_k - t) + s & |s| < |x_k| \end{cases} \\
s &= t
\end{aligned}
$$

and the final result is $s + c$. Every intermediate value stays in
`float32`, but the accumulated correction recovers most of what naive
summation throws away.

## Task

Implement `compensated_sum`:

```python
def compensated_sum(arr: np.ndarray) -> float:
    """Kahan-Neumaier compensated summation, accumulating entirely in
    float32. Returns a Python float."""
```

* `arr` — a 1-D NumPy array of `dtype=float32`.
* Iterate over `arr` element by element with an explicit Python loop (this
  is a hardware-independent precision exercise, not a vectorisation one),
  maintaining `s` and the compensation term `c` as `np.float32` scalars at
  every step, following the update above.
* Return `float(s + c)`.

Do not sidestep the `float32` arithmetic by casting the whole array to
`float64` and calling `np.sum` — the point is to recover precision *within*
`float32`, not to escape it.

## Example

```python
import numpy as np
arr = np.array([1e10, 1.0, 1.0, 1.0, -1e10], dtype=np.float32)
compensated_sum(arr)   # ≈ 3.0 — naive float32 summation returns 0.0 here,
                        # because 1e10 + 1.0 rounds straight back to 1e10
```

## What the gate checks

Two gates must both pass, evaluated on one fixed, shuffled `float32` array
of $3002$ elements engineered so naive summation loses most of its small
increments:

* **improvement_ratio** — the grader computes `naive_rel_err` (plain
  `float32` running-sum error against a `float64` NumPy reference) and
  `student_rel_err` (your result's error against the same reference), then
  reports `naive_rel_err / student_rel_err`. This must be $\ge 100$: your
  compensated sum must be at least 100$\times$ more accurate than naive
  summation.
* **loop_events** — the grader traces Python line execution while your
  function runs (`arena.probe.count_line_events`) and requires at least
  $3000$ line events, confirming the array was actually walked element by
  element rather than reduced with a single vectorised/float64 shortcut.
