## Context

Matrix multiplication computes

$$C_{ij} = \sum_{k=0}^{n-1} A_{ik} B_{kj} .$$

The mathematical result is independent of the loop order, but the memory access pattern is not. For row-major arrays, the loop order changes which addresses are reused before cache lines are evicted.

Consider the three loop nests:

$$
\text{ijk}: \quad i \rightarrow j \rightarrow k
$$

$$
\text{ikj}: \quad i \rightarrow k \rightarrow j
$$

$$
\text{jki}: \quad j \rightarrow k \rightarrow i .
$$

A cache miss occurs when a memory access cannot be served from the configured cache. The total misses depend on the address trace produced by the loop order and the cache parameters. This task models a deterministic cache and ranks the loop orders by simulated misses.

The trace model uses row-major matrices with element size $8$ bytes. Each multiply-add reads one element from $A$, one element from $B$, and updates one element in $C$.

## Task

Implement `rank_matmul_orders()`:

```python
def rank_matmul_orders() -> list[str]:
    ...
```

Return a list containing exactly the three strings `"ijk"`, `"ikj"`, and `"jki"` ordered from lowest simulated cache misses to highest simulated cache misses.

The cache model is fixed by the grader:

- matrix size is $n=24$,
- cache line size is $64$ bytes,
- cache has $32$ sets,
- cache associativity is $2$ ways.

Use the deterministic access model described above. Your implementation should rank the three loop orders by their simulated miss counts, not by measuring real execution time.

## Example

A valid return value has this form:

```python
["ikj", "ijk", "jki"]
```

The exact order depends on the simulated trace and is determined by the cache model.

## What the gate checks

The gate creates the same deterministic traces for all three loop orders and runs them through the cache simulator. It computes the reference ranking from the simulator itself.

Your returned permutation must exactly match the simulator-derived ordering. Real hardware timing, platform cache behaviour, and manually entered expected rankings are not used.
