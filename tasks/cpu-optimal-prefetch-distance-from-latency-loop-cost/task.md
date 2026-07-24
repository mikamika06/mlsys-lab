## Context

A software prefetch instruction attempts to start a memory transfer before the
data is needed. The useful distance between issuing the prefetch and consuming
the value depends on the memory latency and the amount of computation performed
per loop iteration.

If the memory latency is $L$ cycles and one loop iteration costs $C$ cycles,
the number of iterations that must be moved ahead is

$$d = \left\lceil \frac{L}{C} \right\rceil .$$

A distance smaller than this can leave the processor waiting for memory. A
larger distance can waste cache capacity by bringing data in too early. This
task uses a deterministic cache simulator to model the effect of the generated
distance on a fixed access stream.

## Task

Implement `optimal_prefetch_distance(mem_latency, loop_body_cycles)`:

```python
def optimal_prefetch_distance(mem_latency: int, loop_body_cycles: int) -> int:
    ...
```

Return the prefetch distance as an integer. The inputs are positive cycle
counts. The result must be the smallest integer distance $d$ satisfying

$$d \cdot \text{loop\_body\_cycles} \ge \text{mem\_latency}.$$

The grader also builds deterministic byte-address traces from candidate
distances and evaluates them with a fixed cache model. The cache model is used
as an oracle for the simulated reuse behaviour, not for measuring real machine
performance.

## Example

```python
distance = optimal_prefetch_distance(120, 30)
# distance == 4

distance = optimal_prefetch_distance(121, 30)
# distance == 5
```

## What the gate checks

The gate recomputes the mathematical reference distance using integer arithmetic
and verifies an exact match. It also runs deterministic cache simulations for
the generated access patterns using fixed cache parameters. A solution passes
only when its returned distance matches the oracle used by the simulator.
