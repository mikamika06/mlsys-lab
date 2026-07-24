## Context

In disaggregated inference systems, a request can move the key-value (KV) cache between compute resources. The transferred KV bytes depend on the number of layers, the number of KV heads, the attention head dimension, the sequence length, and the storage dtype size.

For a sequence length $s$, the KV transfer size is

$$
B_{\mathrm{KV}} = 2 \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{kv\_heads}} \cdot d_{\mathrm{head}} \cdot s \cdot b_{\mathrm{dtype}},
$$

where the factor $2$ accounts for keys and values.

A system can compare transfer cost against recomputing attention state. Assume recomputation requires

$$
F_{\mathrm{recompute}} =
2 \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{q\_heads}} \cdot d_{\mathrm{head}} \cdot s^2
$$

FLOPs. If each FLOP has a cost of $r$ transferred bytes per FLOP, the recomputation byte equivalent is $rF_{\mathrm{recompute}}$. The break-even sequence length satisfies

$$
B_{\mathrm{KV}} = rF_{\mathrm{recompute}}.
$$

Solving for $s$ gives the sequence length where moving KV and recomputing have equal byte cost.

## Task

Implement `kv_transfer_analysis(config, bytes_per_flop)`:

```python
def kv_transfer_analysis(config: dict, bytes_per_flop: float) -> tuple[int, float]:
    ...
```

The `config` dictionary contains:

- `n_layers`: number of transformer layers
- `n_kv_heads`: number of KV attention heads
- `n_q_heads`: number of query attention heads
- `head_dim`: attention head dimension
- `seq_len`: current request sequence length
- `dtype_bytes`: bytes per stored KV element

Return:

1. `transfer_bytes`: the exact integer KV transfer size for `seq_len`.
2. `break_even_seq_len`: the floating point sequence length where KV transfer and recomputation have equal byte cost.

Use the formulas from the context. Do not round the break-even value.

## Example

```python
config = {
    "n_layers": 32,
    "n_kv_heads": 8,
    "n_q_heads": 32,
    "head_dim": 128,
    "seq_len": 4096,
    "dtype_bytes": 2,
}

transfer, threshold = kv_transfer_analysis(config, 0.25)

# transfer == 536870912
# threshold is the sequence length where transfer and recomputation cost match
```

## What the gate checks

The gate computes the transfer byte count and break-even sequence length independently from the formulas above.

The `exact_match` metric requires the returned transfer byte count to exactly match the oracle integer. The `rel_err` metric requires the returned break-even sequence length to satisfy

$$
\frac{|x_{\mathrm{student}} - x_{\mathrm{oracle}}|}{|x_{\mathrm{oracle}}| + 10^{-12}} \leq 10^{-6}.
$$

The grader uses multiple transformer configurations, including grouped-query attention cases where the number of KV heads differs from the number of query heads.
