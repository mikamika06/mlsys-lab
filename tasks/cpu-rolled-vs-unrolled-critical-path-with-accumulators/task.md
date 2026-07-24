## Context

Loop unrolling reduces loop overhead and can expose instruction-level parallelism. A rolled reduction loop updates one accumulator:

$$
x_{i+1} = x_i + a_i
$$

so each addition depends on the previous addition. If the addition latency is $L$ cycles, a reduction of $n$ elements has a modeled critical path:

$$
C_{\mathrm{rolled}} = nL .
$$

A version using $k$ independent accumulators creates multiple dependency chains:

$$
C_{\mathrm{unrolled}} = \left\lceil \frac{n}{k} \right\rceil L .
$$

The task also models memory behavior by returning the byte addresses touched by the kernel. A deterministic cache simulator evaluates the access trace using fixed cache parameters. Sequential access to contiguous 8-byte elements gives predictable cache reuse because multiple elements share a cache line.

## Task

Implement `model_kernel(n, k)`:

```python
def model_kernel(n: int, k: int):
    ...
```

Return a tuple:

```python
(rolled_critical_path, unrolled_critical_path, addresses)
```

The contract is:

- The addition latency is $L = 4$ cycles.
- The array contains $n$ elements.
- Each element occupies $8$ bytes.
- Element $i$ is located at byte address $8i$.
- `addresses` must contain exactly the addresses read by the kernel in increasing order.

The returned critical paths must use the formulas above. The returned addresses must represent a sequential scan of the input array.

## Example

```python
rolled, unrolled, addrs = model_kernel(8, 4)

# rolled == 32
# unrolled == 8
# addrs == [0, 8, 16, 24, 32, 40, 48, 56]
```

## What the gate checks

The grader computes the critical-path model independently and compares the returned pair exactly.

The grader also runs the returned address trace through a deterministic cache simulator. The trace must produce the same miss behavior as the computed sequential reference trace. The simulator uses fixed parameters: 64-byte cache lines, 16 sets, and 2-way associativity. The check is deterministic and does not use wall-clock measurements.
