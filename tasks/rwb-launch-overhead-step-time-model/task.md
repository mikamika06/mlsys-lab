## Context

Every GPU kernel launch, even a tiny one, pays a fixed **host-side launch
overhead** $L$ (CPU time to set up and enqueue the kernel) on top of
whatever the kernel actually does on the device. A model's forward step
usually issues many kernels — one per op in the graph. **Eager** execution
launches each of the $N$ kernels separately, so the CPU pays $L$ a total
of $N$ times over the step:

$$
T_{\text{eager}} = N \cdot L + C
$$

where $C$ is the total device compute time for the step (assumed
independent of how the kernels were launched — the GPU still has to do
the same work either way). A **CUDA graph** captures the whole sequence of
$N$ kernels once and replays it with a *single* launch, so the host only
pays $L$ once per step regardless of $N$:

$$
T_{\text{graph}} = L + C
$$

The fraction of the step's time that graph capture removes is

$$
f = \frac{T_{\text{eager}} - T_{\text{graph}}}{T_{\text{eager}}} = \frac{(N-1) \cdot L}{N \cdot L + C}
$$

which is large when many small kernels dominate the step ($C \ll N L$) and
small when the step is compute-bound ($C \gg N L$) — graph capture is a
launch-overhead optimization, not a compute optimization.

## Task

Implement `graph_launch_step_time(L, N, C)`:

```python
def graph_launch_step_time(L: float, N: int, C: float) -> np.ndarray:
    ...
```

- `L`: per-kernel launch overhead.
- `N`: number of kernels issued per step.
- `C`: total device compute time per step.

Return `np.array([eager_time, graph_time, fraction_removed])` using the
three formulas above.

## Example

```python
graph_launch_step_time(L=0.1, N=50, C=0.5)
# eager_time = 50*0.1 + 0.5 = 5.5
# graph_time = 0.1 + 0.5 = 0.6
# fraction_removed = (5.5 - 0.6) / 5.5 = 0.8909...
# -> array([5.5, 0.6, 0.89090909])
```

## What the gate checks

The gate evaluates a handful of hand-picked `(L, N, C)` triples spanning a
single-kernel step (no possible savings), a launch-overhead-dominated step
with thousands of tiny kernels, and a compute-dominated step, plus several
randomly generated triples from a seeded generator. For each one it
recomputes the reference `[eager_time, graph_time, fraction_removed]`
directly from the formulas with NumPy and compares it to your output with
relative L2 error, requiring `rel_err < 1e-9`. A solution that computes
`fraction_removed` as `(N * L) / T_eager` — treating *all* launch
overhead as eliminated and forgetting the graph still pays `L` once per
step — instead of `(T_eager - T_graph) / T_eager`, will be close for
large $N$ (where that single remaining launch is a negligible slice) but
measurably wrong whenever $N$ is small enough that it isn't.
