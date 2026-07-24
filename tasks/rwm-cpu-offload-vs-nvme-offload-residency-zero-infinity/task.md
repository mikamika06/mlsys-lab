## Context

Large model training systems can move different parts of model state between memory tiers. In a ZeRO-Infinity style design, parameters and optimizer states may live on the GPU, CPU memory, or NVMe storage.

Assume a model has $N$ parameters. Let each parameter occupy $b_p$ bytes and let optimizer state occupy $b_o$ bytes per parameter. The total memory required for parameters is

$$M_p = N b_p,$$

and the total memory required for optimizer state is

$$M_o = N b_o.$$

The residency policy determines which device owns each portion of the state. GPU memory contains states that are not offloaded, CPU memory contains CPU-offloaded states, and NVMe contains NVMe-offloaded states.

## Task

Implement `zero_residency`:

```python
def zero_residency(
    offload_optimizer: str,
    offload_param: str,
    num_params: int,
    param_bytes: int = 2,
    optimizer_bytes_per_param: int = 12,
) -> dict:
    ...
```

Return a dictionary with exactly these integer keys:

```python
{
    "gpu": ...,
    "cpu": ...,
    "nvme": ...
}
```

The function must compute the byte residency split.

Rules:

- Parameters stay on GPU unless `offload_param == "nvme"`.
- Optimizer states stay on GPU unless `offload_optimizer == "cpu"` or `offload_optimizer == "nvme"`.
- The only valid offload values are `"none"`, `"cpu"`, and `"nvme"`.
- The returned values are total bytes in each location.

## Example

```python
zero_residency(
    offload_optimizer="cpu",
    offload_param="none",
    num_params=1000,
    param_bytes=2,
    optimizer_bytes_per_param=12,
)
```

returns:

```python
{
    "gpu": 2000,
    "cpu": 12000,
    "nvme": 0,
}
```

## What the gate checks

The gate compares the returned residency split against an oracle that independently computes parameter and optimizer byte placement from the residency rules.

The `exact_match` score must be $1.0$ for all tested configurations. The oracle checks GPU, CPU, and NVMe byte totals for CPU optimizer offload and NVMe optimizer plus parameter offload cases.
