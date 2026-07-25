## Context

A modern out-of-order processor can have multiple cache misses **in flight**
simultaneously — a non-blocking cache lets several independent loads stay
outstanding at once instead of stalling the pipeline one at a time. This
capability is called **memory-level parallelism** (MLP). When `mlp`
independent miss streams overlap, the effective latency the program pays per
miss drops, because the misses are serviced concurrently instead of back to
back.

A simple analytical model for a memory-bound loop's stall time is

$$
T_{\text{model}} \;=\; \frac{N_{\text{misses}}}{\text{MLP}} \;\times\; t_{\text{miss}}
$$

where $N_{\text{misses}}$ is the total number of cache-line misses the loop
produces, `MLP` is the number of outstanding misses the hardware can service
in parallel, and $t_{\text{miss}}$ is the latency of a single miss in cycles.
The intuition: `N_misses` misses, grouped into waves of up to `MLP`
concurrent ones, take `N_misses / MLP` waves, each wave costing one
`miss_latency`.

## Task

Implement

```cpp
double modeled_cycles(long num_misses, int mlp, double miss_latency);
```

which returns $T_{\text{model}} = (\text{num\_misses} / \text{mlp}) \times \text{miss\_latency}$,
computed in floating point (do not truncate the division to an integer
first).

The fixed driver (`main.cpp`) produces `num_misses` itself, for five
scenarios, by replaying a real strided access trace (`n_nodes` elements
spaced `node_size` bytes apart — a pointer-chase-shaped pattern) through a
deterministic set-associative LRU cache model parameterized by
`line_bytes`/`sets`/`ways`; your function only has to turn the resulting
miss count, `mlp`, and `miss_latency` into a cycle count.

## Example

```
num_misses = 256, mlp = 4, miss_latency = 200.0
modeled_cycles(256, 4, 200.0) -> (256 / 4) * 200.0 = 12800.0
```

A model that instead serializes every miss —
`num_misses * miss_latency = 256 * 200.0 = 51200.0` — ignores memory-level
parallelism entirely and overestimates the true stall time by exactly a
factor of `mlp`.

## What the gate checks

`main.cpp` runs five fixed `(n_nodes, node_size, line_bytes, sets, ways,
mlp, miss_latency)` scenarios, computes each real miss count from its own
cache model, calls `modeled_cycles` on it, and prints `misses=<N>
cycles=<T>` per scenario. `verify_native.sh` compiles `solve.cpp` and
`ref.cpp` against the same `main.cpp` with `clang++ -O2 -std=c++20`, runs
both, and requires

$$
\max_k \lvert T_{\text{candidate}}^{(k)} - T_{\text{ref}}^{(k)} \rvert \le 10^{-6}
$$

Since both binaries share the same driver and cache model, the printed
`misses=<N>` values always agree — only a wrong `modeled_cycles` formula
(forgetting to divide by `mlp`, dividing by the wrong quantity, swapping the
multiply for an add, ...) can make the printed `cycles=<T>` values diverge
from the reference.
