## Context

A tensor that FITS in some level of the cache hierarchy only has to be
loaded from DRAM **once**, no matter how many times the layer reads it
afterward — every later read is served from the cache. A tensor that
DOESN'T fit in even the largest cache level has nowhere to stay resident:
every one of its reads goes all the way back to DRAM. For a transformer
layer's DRAM traffic budget, the difference between "read once, resident"
and "read `num_uses` times, streamed" is the whole story — and it depends
entirely on whether `tensor_bytes <= cache_capacity`, not on how many
FLOPs the tensor is involved in.

## Task

Implement

```cpp
long classify_layer_residency(const long* tensor_bytes, const int* num_uses, int num_tensors,
                               const long* cache_capacities, int num_levels, int* residency_out);
```

`cache_capacities[0..num_levels)` is strictly increasing (level 0
smallest/fastest). For every tensor `i`, find the smallest level `L` with
$\text{tensor\_bytes}[i] \le \text{cache\_capacities}[L]$ and write
`residency_out[i] = L`; if no such level exists (the tensor exceeds even
`cache_capacities[num_levels-1]`), write `residency_out[i] = -1`
(streamed). Return the total modeled DRAM byte budget:

$$
\text{budget} = \sum_i \begin{cases}
\text{tensor\_bytes}[i] & \text{residency\_out}[i] \ne -1 \text{ (resident)} \\
\text{tensor\_bytes}[i] \times \text{num\_uses}[i] & \text{residency\_out}[i] = -1 \text{ (streamed)}
\end{cases}
$$

## Example

`tensor_bytes=4194304` (a 4 MiB weight matrix), `num_uses=2`,
`cache_capacities=[49152, 1572864, 3145728]` (48 KiB / 1.5 MiB / 3 MiB):
4194304 exceeds even the largest level (3145728), so it's streamed —
`residency_out[i] = -1` and it costs `4194304 * 2 = 8388608` bytes of
DRAM traffic, not `4194304`.

## What the gate checks

`main.cpp` runs two fixed scenarios. Scenario 1 models one real
attention+MLP layer (`d_model=512, n_heads=8, seq_len=128, d_ff=4096`) —
12 tensors (QKVO weights, Q/K/V-projections, attention scores, attention
output, the two MLP weight matrices, and the MLP hidden activation)
against 48 KiB / 1.5 MiB / 3 MiB cache levels; the two 4 MiB MLP weight
matrices don't fit any level and are each read twice (the layer processes
the sequence in 2 chunks). Scenario 2 is a tiny 4-tensor layer where
everything fits the smallest cache level, so the budget reduces to a
plain sum of tensor sizes regardless of `num_uses`. Both scenarios print
the residency vector and the total budget. The candidate's full stdout is
compared byte-for-byte (`exact_match = 1.0`) against the reference's. On
scenario 1 the reference gets `residency = [1,1,1,1,1,1,1,1,1,-1,-1,1]`
and `budget=20971520` — charging the two streamed weight matrices for
**both** of their reads (`4194304*2` each) rather than one accounts for
16777216 of that total; treating every tensor as resident after one load
(ignoring `num_uses` for the streamed ones) would undercount the budget
by exactly that much.
