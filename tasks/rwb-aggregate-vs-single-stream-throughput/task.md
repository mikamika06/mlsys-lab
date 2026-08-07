## Context

In a parallel processing system we often have $N$ independent slots that can each hold at most one token per time step.  
Let $\mathbf{T}\in\{0,1\}^{S\times N}$ be the occupancy trace where $S$ is the number of steps and $T_{ij}=1$ if slot $j$ is busy at step $i$.  

Define the **tokens per step** vector
$$
t_i = \sum_{j=1}^N T_{ij}\,,
$$
and the **aggregate throughput**
$$
\mathcal{T}_{\text{agg}}=\frac{1}{S}\sum_{i=1}^{S} t_i\,.
$$

If we had only a single stream that could occupy one slot at a time, the best achievable rate is one token per step whenever *any* slot is busy.  The corresponding **single‑stream rate** is
$$
\mathcal{T}_{\text{ss}}=\frac{1}{S}\sum_{i=1}^{S} \mathbf{1}_{t_i>0}\,,
$$
where $\mathbf{1}_{\cdot}$ is the indicator function.  This quantity lies in $[0,1]$ and measures how often the system is busy.

The task below asks you to compute both $\mathcal{T}_{\text{agg}}$ and $\mathcal{T}_{\text{ss}}$ from a list trace without using explicit Python loops.

## Task

Implement the function `throughput(trace)`:

```python
def throughput(trace: list[list[int]]) -> list[float]:
    ...
```

* `trace` is a 2‑D integer list of shape `(S, N)` with entries in `{0,1}`.  
* The function must return a one‑dimensional float64 array `[agg, single]` where
  * `agg`   = $\mathcal{T}_{\text{agg}}$,
  * `single`= $\mathcal{T}_{\text{ss}}$.

The implementation should be fully vectorized; no Python `for` loops are allowed. The result must have dtype `float64`.

## Example

```python
trace = [[1,0,0],
                  [1,1,0],
                  [0,0,0]]
# tokens per step: [1, 2, 0]
# aggregate throughput: (1+2+0)/3 = 1.0
# single‑stream rate:   (1 + 1 + 0)/3 = 0.666...
out = throughput(trace)
print(out)          # array([1.        , 0.66666667])
```

## What the gate checks

The grader computes a Python reference implementation and compares your output with it using the global relative L2 error
$$
\mathrm{rel\_err}=\frac{\lVert \hat y - y\rVert}{\lVert y\rVert}\,.
$$
Your solution must achieve $\mathrm{rel\_err}\le 10^{-9}$ on a set of random traces. The gate metric is `rel_err`. No timing or line‑event constraints are imposed; the focus is correctness.
