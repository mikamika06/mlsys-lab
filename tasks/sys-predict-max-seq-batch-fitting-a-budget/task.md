## Context

Autoregressive language models store previous key and value tensors in a KV cache so that
new tokens do not require recomputing attention states. For a transformer with $L$ layers,
$H_{kv}$ key/value heads, head dimension $d$, and element size $b$ bytes, the cache bytes
for batch size $B$ and sequence length $S$ are modeled as

$$
M_{kv} = 2 \cdot L \cdot B \cdot S \cdot H_{kv} \cdot d \cdot b .
$$

The factor $2$ accounts for storing both keys and values. The available memory budget must
also leave room for fixed model memory. If the total budget is $M$ bytes and the model uses
$M_{fixed}$ bytes, the KV cache must satisfy

$$
M_{kv} \leq M - M_{fixed}.
$$

For serving systems, a scheduler often needs boundary points showing the largest sequence
length that fits for each batch size.

## Task

Implement `predict_max_seq_batch(config)`.

The function receives a dictionary with integer fields:

- `budget_bytes`: total memory budget in bytes.
- `fixed_bytes`: memory used by the model and other allocations.
- `layers`: number of transformer layers $L$.
- `kv_heads`: number of key/value heads $H_{kv}$.
- `head_dim`: key/value head dimension $d$.
- `bytes_per_element`: bytes per cache element $b$.
- `max_batch`: largest batch size to consider.
- `max_seq`: largest sequence length to consider.

Return a list of dictionaries. For every batch size $B$ from $1$ through `max_batch`, include:

```python
{"batch": B, "max_seq": S}
```

where $S$ is the largest integer sequence length from $1` through `max_seq` satisfying the
KV-cache budget constraint. If no sequence length fits, use `0`.

The output list must be ordered by increasing batch size.

## Example

```python
config = {
    "budget_bytes": 100000,
    "fixed_bytes": 10000,
    "layers": 2,
    "kv_heads": 2,
    "head_dim": 4,
    "bytes_per_element": 2,
    "max_batch": 3,
    "max_seq": 100,
}

predict_max_seq_batch(config)

# [
#   {"batch": 1, "max_seq": 100},
#   {"batch": 2, "max_seq": 100},
#   {"batch": 3, "max_seq": 93},
# ]
```

## What the gate checks

The gate builds a reference result by evaluating the byte-accurate KV-cache model and
compares the returned boundary points with an exact match check. Implementations that
ignore the two cache tensors, use the wrong head count, or return approximate values fail
the gate.
