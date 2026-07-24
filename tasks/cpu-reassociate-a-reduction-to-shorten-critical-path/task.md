## Context

A reduction such as summing an array can be limited by the dependency chain between additions. A left-associated reduction

$$
((((a_0+a_1)+a_2)+a_3)+\dots)+a_{n-1}
$$

has a critical path of $O(n)$ additions because every addition depends on the previous result.

A balanced reduction tree reassociates the operations:

$$
(a_0+a_1) + (a_2+a_3) + \dots
$$

and combines partial sums in levels. The number of dependent levels becomes approximately

$$
\lceil \log_2(n) \rceil ,
$$

which exposes more instruction-level parallelism.

This task models the reduction tree rather than measuring real CPU execution. The returned schedule describes which input addresses are read and how many dependency levels are required.

## Task

Implement `reassociated_sum_trace(values, base_addr)`:

```python
def reassociated_sum_trace(values: list[float], base_addr: int) -> dict:
    ...
```

Return a dictionary with exactly these keys:

- `total`: the sum of all values as a Python `float`.
- `addrs`: a list of byte addresses in the order the reduction reads inputs. Each element of `values` is stored as an 8-byte value starting at `base_addr`.
- `critical_path`: the number of addition levels in the reduction tree.

The reduction must be reassociated as a balanced binary tree. The address trace should visit each input exactly once in increasing index order. The critical path for $n$ values is the height of the balanced reduction tree, computed by repeatedly combining pairs until one value remains.

Do not use timing or hardware-specific features. The returned trace is a deterministic model of the optimized reduction.

## Example

```python
values = [1.0, 2.0, 3.0, 4.0]
result = reassociated_sum_trace(values, 4096)

# result["total"] == 10.0
# result["addrs"] == [4096, 4104, 4112, 4120]
# result["critical_path"] == 2
```

## What the gate checks

The gate recomputes the expected balanced reduction model and checks the returned result exactly.

The harness also sends the emitted address trace through a deterministic cache simulator with fixed cache parameters. The cache miss count must match the reference access pattern. The simulator is used as the oracle, not wall-clock timing or real hardware measurements.
