## Context

Every kernel launch pays a fixed overhead `launch_cost` (queueing the
launch, setting up the grid, whatever else doesn't scale with how much
work the kernel does) before a single element gets processed. Launching
many SMALL kernels to do a job pays that overhead every single time;
launching ONE kernel over the whole workload pays it once. The
overhead-per-element for a launch covering `N` elements is
`launch_cost / N` — it shrinks as `N` grows, but never disappears.

The crossover point is where that shrinking overhead-per-element finally
drops to (or below) what a single element's own compute already costs
(`per_elem_cost`): below the crossover, overhead dominates and you're
mostly paying for the launch, not the work; at or above it, the launch's
fixed cost is amortized into insignificance relative to the compute
itself.

$$N^* = \left\lceil \frac{\text{launch\_cost}}{\text{per\_elem\_cost}} \right\rceil$$

## Task

Write a CUDA-C kernel (single thread — this derives one number from two
scalars):

```cpp
__global__ void crossover_n(float* out, float launch_cost, float per_elem_cost);
```

`out[0] = ceilf(launch_cost / per_elem_cost)`.

## Example

| launch_cost | per_elem_cost | $N^*$ |
|---|---|---|
| 1000 | 5  | $\lceil 200.0 \rceil = 200$ |
| 250  | 3  | $\lceil 83.33\ldots \rceil = 84$ |
| 5000 | 50 | $\lceil 100.0 \rceil = 100$ |
| 1    | 7  | $\lceil 0.142857\ldots \rceil = 1$ |

Cheap per-element work relative to a large launch overhead (`1000` vs
`5`) needs a big batch — `200` elements — before the launch is worth it
on its own; when per-element cost already dwarfs the launch overhead
(`1` vs `7`), even a single element clears the crossover immediately.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU once per fixed `(launch_cost, per_elem_cost)`
case, requiring `max_abs_err <= 1e-6` against
`math.ceil(launch_cost / per_elem_cost)`. Using plain division without
the ceiling (returning `83.333...` instead of `84` for the second case)
or dividing the arguments in the wrong order both produce numbers well
outside tolerance for at least one of the four fixed cases. The empty
starter leaves the output at its `-1.0` sentinel.
