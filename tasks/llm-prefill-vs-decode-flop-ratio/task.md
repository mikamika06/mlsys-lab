## Context

A decoder-only transformer serves a request in two phases. **Prefill** runs the
whole prompt of $P$ tokens through one forward pass (no KV cache yet). **Decode**
then generates tokens one at a time: each step feeds a *single* new token but lets
it attend to all $T$ keys already in the KV cache. The two phases stress the
hardware very differently, and the cleanest way to see why is to count the
floating-point work in one decoder layer.

We count only the matrix-multiply FLOPs, using the standard convention that a
matmul $A_{m\times k} B_{k\times n}$ costs $2mkn$ FLOPs (one multiply + one add
per inner product term). The layer has hidden size $d$ (`d_model`), an
$h$-head attention block ($h$ = `n_heads`, head dim $d_h = d/h$), and a
feed-forward block with inner width $f$ (`d_ff`). Its projections are
$W_Q,W_K,W_V,W_O \in \mathbb{R}^{d\times d}$, $W_{up}\in\mathbb{R}^{d\times f}$,
$W_{down}\in\mathbb{R}^{f\times d}$.

**Prefill** ($P$ tokens, causal self-attention over $P$ positions):

$$
\text{FLOPs}_{\text{prefill}}
= \underbrace{8\,P\,d^{2}}_{Q,K,V,O\text{ projections}}
+ \underbrace{4\,P^{2}\,d}_{QK^{\top}\text{ and }AV}
+ \underbrace{4\,P\,d\,f}_{\text{FFN up + down}} .
$$

**Decode** (1 new token attending to $T$ cached keys/values):

$$
\text{FLOPs}_{\text{decode}}
= \underbrace{8\,d^{2}}_{Q,K,V,O\text{ projections}}
+ \underbrace{4\,T\,d}_{qK^{\top}\text{ and }aV}
+ \underbrace{4\,d\,f}_{\text{FFN up + down}} .
$$

Note the head split does not change the totals: summing $2P^{2}d_h$ over the
$h$ heads gives $2P^{2}d$ regardless of $h$. The ratio
$\text{FLOPs}_{\text{prefill}} / \text{FLOPs}_{\text{decode}}$ is why prefill is
compute-bound while decode is memory-bound: prefill does roughly $P\times$ the
projection work of a single decode step.

## Task

Implement `prefill_vs_decode_flops(d_model, n_heads, d_ff, P, T)`:

```python
def prefill_vs_decode_flops(d_model: int, n_heads: int, d_ff: int, P: int, T: int) -> dict:
    ...
```

Return a dict with three keys:

- `"prefill"` — integer matmul FLOPs of one prefill pass over $P$ tokens,
- `"decode"` — integer matmul FLOPs of one decode step attending to $T$ keys,
- `"ratio"` — the float ratio `prefill / decode`.

Count matmul FLOPs only (each $A_{m\times k}B_{k\times n}$ contributes $2mkn$).
Cached keys/values are already in memory, so only the new token's $K,V$
projections are computed during decode.

## Example

```python
prefill_vs_decode_flops(d_model=64, n_heads=8, d_ff=256, P=16, T=32)
# prefill = 8*16*64^2 + 4*16^2*64 + 4*16*64*256 = 524288 + 65536 + 1048576 = 1638400
# decode  = 8*64^2   + 4*32*64   + 4*64*256     =  32768 +  8192 +  65536 =  106496
# -> {"prefill": 1638400, "decode": 106496, "ratio": 15.384615384615385}
```

## What the gate checks

Two gates. The grader builds the actual layer weights and runs a real NumPy
prefill and decode forward pass for each configuration, tallying `2*m*k*n` for
every matmul it executes — that execution-derived count is the oracle. The
`exact_match` gate requires your integer `prefill` and `decode` counts to equal
the oracle for every case; `ratio_rel_err` requires your `ratio` field to match
`prefill / decode` to a relative error of $10^{-9}$. There is no formula
hardcoded in the grader, so only a correct accounting of every matmul passes.
