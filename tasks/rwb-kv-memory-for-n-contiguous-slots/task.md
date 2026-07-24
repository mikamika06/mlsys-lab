## Context

In transformer models each layer maintains a key/value (KV) cache that stores the hidden states of all tokens processed so far.  
When the model is run in *contiguous* mode – i.e., without paging or recycling – every slot in the cache must hold a full KV tensor for the entire context length $n_{\text{ctx}}$.  

For a single layer the size of one KV tensor (key or value) is

$$
\text{size}_{\text{KV}} = \text{kv\_heads} \times \text{head\_dim} \times n_{\text{ctx}} \times \text{dtype\_bytes},
$$

where $\text{dtype\_bytes}$ is the number of bytes per element (e.g. $4$ for `float32`).  
Because a layer stores both keys and values, the memory required by one slot for all layers is

$$
\text{slot\_bytes} = 2 \times \text{layers}
\times \text{kv\_heads}
\times \text{head\_dim}
\times n_{\text{ctx}}
\times \text{dtype\_bytes}.
$$

If the model reserves $N$ contiguous slots, the total KV memory is simply

$$
\boxed{\text{total\_bytes} = N \times \text{slot\_bytes}}.
$$

The task is to implement this calculation as a pure Python function.

## Task

Implement `kv_memory_bytes(cfg: dict, n_slots: int) -> int`:

```python
def kv_memory_bytes(cfg: dict, n_slots: int) -> int:
    ...
```

`cfg` contains the keys:

- `"layers"` – number of transformer layers (int)
- `"kv_heads"` – number of KV heads per layer (int)
- `"head_dim"` – dimensionality of each head (int)
- `"dtype_bytes"` – bytes per element in the KV tensors (int)
- `"n_ctx"` – context length, i.e. number of tokens stored per slot (int)

The function must return the total number of bytes required to store $N$ contiguous slots.

## Example

```python
cfg = {
    "layers": 12,
    "kv_heads": 12,
    "head_dim": 64,
    "dtype_bytes": 4,   # float32
    "n_ctx": 2048
}
print(kv_memory_bytes(cfg, 3))
# 3 * 2 * 12 * 12 * 64 * 2048 * 4 = 179\,229\,120
```

## What the gate checks

The grader verifies that your implementation returns exactly the integer value computed by the reference formula for a set of test cases. No floating‑point tolerance is needed because all arithmetic involves integers.
