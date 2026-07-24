## Context

ZeRO partitions model states across data-parallel ranks. A closed-form memory estimate starts from the total model bytes and divides partitioned components across ranks.

Let:
- $P$ be the number of model parameters.
- $N$ be the number of data-parallel ranks.
- $s$ be the number of bytes per parameter.
- $S$ be the ZeRO stage.

The unpartitioned component sizes are:

$$
\text{param\_bytes} = P s
$$

$$
\text{grad\_bytes} = P s
$$

$$
\text{optimizer\_bytes} = 2 P s
$$

For ZeRO stage 1, optimizer states are partitioned. For stage 2, optimizer states and gradients are partitioned. For stage 3, optimizer states, gradients, and parameters are partitioned.

Before offload, the per-rank GPU memory estimate is:

$$
M_{\mathrm{gpu}} =
\text{local unpartitioned bytes}
+
\frac{\text{partitioned bytes}}{N}.
$$

Offload moves selected components from GPU memory to another device. The offloaded components must be subtracted from the GPU total and added to the off-device total.

The offload target can contain optimizer states, parameters, or both. Stage 1 and 2 do not allow parameter offload in this simplified model because parameters remain replicated.

## Task

Implement `zero_rank_bytes(Phi, N, stage, offload_target)`:

```python
def zero_rank_bytes(Phi: int, N: int, stage: int, offload_target: str) -> tuple[int, int]:
    ...
```

Return `(gpu_bytes, offload_bytes)` for one rank.

Arguments:
- `Phi` is the parameter count.
- `N` is the data-parallel world size.
- `stage` is one of `1`, `2`, or `3`.
- `offload_target` is one of `"none"`, `"optimizer"`, `"param"`, or `"optimizer+param"`.

Assume each parameter and gradient element uses $2$ bytes. Each parameter has optimizer state storage of $4$ bytes total.

The function must compute the local GPU bytes after applying the offload subtraction and the total bytes moved off GPU for that rank.

## Example

```python
gpu, off = zero_rank_bytes(1000, 4, 3, "optimizer+param")
# gpu == 5000
# off == 2500
```

## What the gate checks

The gate generates multiple stage, rank, parameter-count, and offload combinations. It computes the expected GPU and off-device byte counts using the same mathematical ZeRO decomposition and checks that the implementation returns an exact tuple match.

A solution that forgets to subtract offloaded components, treats all stages identically, or adds offloaded bytes without removing them from GPU memory will fail.
