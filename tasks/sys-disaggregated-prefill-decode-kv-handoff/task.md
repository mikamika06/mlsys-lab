## Context

In an LLM inference pipeline, the workload is separated into two phases:
the *prefill* phase processes the input prompt in parallel and produces a
*key-value (KV) cache* for each transformer layer; the *decode* phase
autoregressively generates tokens one at a time, leveraging the KV cache to
avoid recomputation. Under a *disaggregated architecture*, prefill and decode
run on separate machines. The KV cache must therefore be transferred from the
prefill node to the decode node over a network.

The KV cache for a layer consists of two tensors, keys and values, each of
shape $(1, H, T, D)$ where $H$ is the number of attention heads, $T$ the
sequence length, and $D$ the head dimension. When the prefill node receives
a long prompt, it may split the prompt into chunks, process each chunk to
produce partial KV tensors, and accumulate the cache incrementally. The decode
node must assemble the final contiguous KV cache from these pieces before it
can start generating.

## Task

Implement the function `assemble_cache(chunks)`. The function accepts a list
of chunks, each describing a contiguous block of the KV cache, and returns the
fully concatenated cache for all layers.

```python
def assemble_cache(chunks: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]) -> list[tuple[np.ndarray, np.ndarray]]:
```

Each element of `chunks` is a tuple `(start_pos, layer_kv)`:

- `start_pos` (int): token-position index of the first token in this chunk
  within the full sequence.
- `layer_kv` is a list of length $L$ (number of layers). The $i$-th element is
  a tuple `(keys, values)` where `keys` and `values` are `np.ndarray` of shape
  $(1, H, C, D)$ ($C$ is this chunk’s length).

The function must return a list of $L$ tuples `(full_keys, full_values)`,
one per layer. The arrays `full_keys` and `full_values` for each layer are
obtained by concatenating the key/value tensors from all chunks **in order of
increasing `start_pos`** along the sequence axis (axis = 2). The total sequence
length

$$T = \sum_{\text{chunks}} C_i$$

must equal the sum of all chunk lengths.

You may assume that the chunks cover the entire sequence exactly (no gaps,
no overlaps) and are already given in order of strictly increasing
`start_pos`. If this assumption is violated the behaviour is undefined.

The `dtype` of the output arrays must match the `dtype` of the input arrays
(commonly `float32`). The implementation must be vectorised: use
`np.concatenate` rather than manual Python indexing.

## Example

```python
import numpy as np

# Single-layer model.
# Chunk 1: start_pos 0, length 2.
k1 = np.array([[[[0.1, 0.2], [0.3, 0.4]]]])  # shape (1,1,2,2)
v1 = np.array([[[[0.5, 0.6], [0.7, 0.8]]]])

# Chunk 2: start_pos 2, length 1.
k2 = np.array([[[[0.9, 1.0]]]])               # shape (1,1,1,2)
v2 = np.array([[[[1.1, 1.2]]]])

chunks = [(0, [(k1, v1)]), (2, [(k2, v2)])]

assembled = assemble_cache(chunks)
print(assembled[0][0])   # full keys
print(assembled[0][1])   # full values
# Keys shape (1,1,3,2) =
# [[[0.1,0.2],
#   [0.3,0.4],
#   [0.9,1.0]]]
# Values shape (1,1,3,2) =
# [[[0.5,0.6],
#   [0.7,0.8],
#   [1.1,1.2]]]
```

## What the gate checks

The gate `max_abs_err` measures the maximum absolute difference between your
assembled cache and the correct concatenation. Every element must satisfy

$$\max_{i,j,k,l} \, |\, \text{your}[i,j,k,l] - \text{reference}[i,j,k,l] \,| \le 10^{-5}.$$

If the output has a wrong number of layers, an incorrect shape, or mis-ordered
concatenation, the error will be well above the threshold and the gate will fail.
