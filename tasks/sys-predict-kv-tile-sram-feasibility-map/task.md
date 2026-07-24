## Context

FlashAttention reduces attention memory pressure by computing attention in tiles that fit in on-chip SRAM. A forward pass processes query, key, and value blocks rather than materializing the full attention matrix.

For a KV tile with sequence length $s$ and head dimension $d$, the simplified SRAM model used in this task stores key and value elements in FP16. The required bytes are

$$
\mathrm{bytes}(s,d) = 2 \cdot s \cdot d \cdot 2 ,
$$

where the first factor accounts for the K and V tensors and the second factor is the number of bytes per FP16 element.

A configuration is feasible when the required SRAM is no larger than the available SRAM budget $B$:

$$
\mathrm{fits}(s,d,B) =
\begin{cases}
\mathrm{True} & \mathrm{bytes}(s,d) \le B \\
\mathrm{False} & \mathrm{otherwise}
\end{cases}
$$

The model intentionally ignores other kernel metadata and focuses on predicting the KV tile capacity boundary.

## Task

Implement `kv_tile_sram_feasibility_map(configs)`:

```python
def kv_tile_sram_feasibility_map(configs):
    ...
```

The input `configs` is a list of tuples. Each tuple is `(seq, d, sram_bytes)`, where:

- `seq` is the KV tile sequence length.
- `d` is the head dimension.
- `sram_bytes` is the available SRAM budget in bytes.

Return a list of booleans with one entry per configuration. The entry must be `True` when the KV tile fits in SRAM and `False` when it spills.

## Example

```python
configs = [
    (64, 64, 16384),
    (128, 128, 65536),
    (256, 128, 65536),
]

result = kv_tile_sram_feasibility_map(configs)
# [True, True, False]
```

## What the gate checks

The gate computes the expected feasibility map using a NumPy-based SRAM budget oracle and compares the returned boolean list exactly.

A single incorrect boundary decision causes the exact match gate to fail. The implementation must correctly apply the KV tile byte model for all tested sequence lengths, dimensions, and SRAM budgets.
