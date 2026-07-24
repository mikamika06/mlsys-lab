## Context

Flash attention never materializes the `S x S` score matrix. It streams keys
and values, keeping a running max `m`, a running denominator `l` and a running
numerator `acc`, and rescales them whenever the max moves:

```
m_new = max(m, score)
corr  = exp(m - m_new)
l     = l * corr + exp(score - m_new)
acc   = acc * corr + exp(score - m_new) * V[j]
```

Causality says query `i` may only attend to keys `j <= i`. The obvious way to
write that is `if (j > i) continue;` — and on real SIMT hardware that's the
expensive way: lanes of the same warp would issue different numbers of
loads, and the warp diverges. The cheap way is **predication**: compute the
score anyway and fold a 0/1 weight into it, so the masked term contributes
nothing while every lane issues identical memory traffic. (This CUDA-C
subset doesn't even have `continue` — predication is the only way to write
the mask here.)

## Task

Implement, as real CUDA-C:

```cpp
__global__ void flash_step(const float* q, const float* k, const float* v, float* o,
                            int s, int d, float scale);
```

One query per lane: `s = 32` (queries = keys = one warp), `d = 4` (head
dim, fixed — this CUDA-C subset has no local arrays, so unroll the
per-dimension accumulators into 4 named scalars). `q`, `k`, `v` are `s x d`
row-major; write `o[i*d + c] = acc_c / l_i` for lane `i`'s query.

Stream `j = 0 .. s-1`. For each `j`, compute the dot product `q[i] . k[j]`,
scale it, then apply the causal mask **by predication**:

```cpp
float keep = (j <= i) ? 1.0 : 0.0;
float score = dot * keep + (-1e30) * (1.0 - keep);
```

— the masked score is so negative it exponentiates to (numerically) zero,
with no branch and no skipped memory access. Then do the usual online-
softmax update (`m_new`, `corr`, `l_i`, and the accumulator) and move on.

## Example

For `j <= i`, `keep = 1.0` and `score` is the real scaled dot product. For
`j > i`, `keep = 0.0` and `score = -1e30`, so `expf(score - m_new)`
contributes essentially nothing to `l_i` or the accumulator — the same
effect as skipping `j`, without ever skipping the loads.

## What the gate checks

- `max_abs_err <= 1e-9` against a causal-softmax-attention oracle computed
  independently in numpy.
- `divergences <= 0`: the simulator counts a warp as diverged when its
  lanes issued different numbers of memory accesses. An early skip on
  `j > i` (even simulated via, say, an `if` that omits the loads) makes
  lane `0`'s inner loop touch far less memory than lane `31`'s — divergence
  greater than zero, and the gate fails even if the printed numbers would
  have been right.
