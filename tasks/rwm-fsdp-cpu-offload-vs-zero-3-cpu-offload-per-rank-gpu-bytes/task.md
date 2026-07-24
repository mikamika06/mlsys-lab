## Context

Fully sharded training schemes reduce per-rank GPU memory by partitioning model state across $N$ ranks. Both FSDP CPU offload and ZeRO-3 CPU offload keep the sharded optimizer and parameter states on CPU, but their temporary GPU residency differs during computation.

Assume a model has $P$ parameters. A parameter element, gradient element, and optimizer state element each use $s$ bytes. Each rank owns a shard of size

$$
S = \frac{P}{N}
$$

elements.

For this simplified memory accounting model:

- FSDP with CPU offload keeps one parameter shard on GPU for computation, then offloads it. It also needs one gradient shard buffer.
- ZeRO-3 with CPU offload gathers full parameters for computation and releases them afterward. It keeps one gradient shard buffer.

The per-rank GPU bytes are therefore:

$$
B_{\mathrm{fsdp}} = 2S s
$$

and

$$
B_{\mathrm{zero3}} = (P + S)s .
$$

The difference is

$$
\Delta B = B_{\mathrm{zero3}} - B_{\mathrm{fsdp}} .
$$

This task uses integer element counts and byte sizes to make the accounting deterministic.

## Task

Implement `compare_offload_bytes(P, N, bytes_per_element)`.

The function receives:

- `P`: total parameter elements, a positive integer.
- `N`: number of data-parallel ranks, a positive integer divisor of `P`.
- `bytes_per_element`: bytes used by one stored tensor element.

Return a dictionary with exactly these integer fields:

```python
{
    "fsdp_gpu_bytes": ...,
    "zero3_gpu_bytes": ...,
    "difference_bytes": ...
}
```

The values must represent the per-rank transient GPU memory for the two CPU-offload schemes described above.

## Example

```python
result = compare_offload_bytes(1000000, 4, 2)

# {
#   "fsdp_gpu_bytes": 1000000,
#   "zero3_gpu_bytes": 2500000,
#   "difference_bytes": 1500000
# }
```

## What the gate checks

The gate generates several parameter configurations and computes the expected values using an independent oracle implementation of the memory accounting formulas.

The returned dictionary must exactly match the oracle output for every case. The gate catches implementations that treat FSDP and ZeRO-3 CPU offload as identical, forget the transient full-parameter gather in ZeRO-3, or report total model memory instead of per-rank GPU bytes.
