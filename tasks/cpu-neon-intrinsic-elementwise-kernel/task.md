## Context

SIMD kernels process multiple independent values per instruction. NEON on Apple silicon and AVX on x86 both use this idea: load a contiguous group of elements, apply the same operation, and store the result.

This task models an elementwise kernel over an array of 32-bit integers. The kernel computes

$$y_i = 3x_i + 7$$

for every element $i$. A vectorized implementation should prefer sequential memory access because hardware caches transfer data in cache lines. The access trace of a kernel is a sequence

$$A = (a_0, a_1, \dots, a_{m-1})$$

where each $a_i$ is a byte address touched by the kernel.

The deterministic cache simulator uses fixed parameters. A cache line contains `line_bytes` bytes, with `sets` independent sets and `ways` lines per set. The simulator determines misses from the access trace, so the gate measures memory behaviour without relying on real CPU timing.

## Task

Implement `elementwise_kernel(n)`:

```python
def elementwise_kernel(n: int) -> tuple[bytes, list[int]]:
    ...
```

Return a pair `(output_bytes, access_trace)`.

`output_bytes` must contain the little-endian 32-bit signed integer results for the input sequence

$$x_i = i \quad \text{for} \quad 0 \le i < n.$$

Therefore the output contains

$$y_i = 3i + 7.$$

`access_trace` must contain the byte addresses touched by the kernel. Model the input array as starting at address `0` and the output array as starting at address `4096`. Each integer occupies $4$ bytes. The trace must describe reads from the input and writes to the output. A cache-friendly SIMD kernel processes elements in increasing index order.

## Example

```python
out, trace = elementwise_kernel(3)

# out contains:
# [7, 10, 13] encoded as little-endian int32 bytes

# A valid sequential trace starts with:
# [
#   0, 4, 8,
#   4096, 4100, 4104
# ]
```

## What the gate checks

The `byte_exact_fraction` score compares the returned bytes with a reference kernel that computes the same elementwise operation.

The `cache_miss_count` score runs the returned access trace through a deterministic cache simulator. The reference cache configuration is fixed by the grader. A low miss count requires a contiguous access pattern instead of repeatedly revisiting scattered addresses.

The grader computes its own reference output and cache simulation results. It does not use wall-clock measurements or hardware-specific CPU behaviour.
