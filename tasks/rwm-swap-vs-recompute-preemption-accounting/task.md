## Context

When a GPU's KV cache is full and a new sequence must be admitted, an LLM serving engine such as vLLM **preempts** a victim sequence. Two strategies exist:

1. **Swap** — copy the victim's KV-cache blocks from GPU to CPU memory; copy them back when the victim resumes.
2. **Recompute** — discard the victim's KV cache entirely; re-prefill all of its tokens from scratch when it resumes.

The KV cache for a single token spans $L$ transformer layers, $H_{\text{kv}}$ KV heads, and $d_{\text{head}}$ dimensions. With a dtype element size of $b$ bytes the per-token KV footprint is

$$s_{\text{token}} = L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b \text{ bytes}.$$

KV cache is managed in fixed-size **blocks** of $B$ tokens each. A sequence of length $T$ occupies

$$N_{\text{blocks}} = \lceil T / B \rceil \text{ blocks},$$

each consuming $B \cdot s_{\text{token}}$ bytes of contiguous memory.

**Swap cost** counts total bytes transferred in the round-trip (GPU→CPU + CPU→GPU):

$$C_{\text{swap}} = 2 \cdot N_{\text{blocks}} \cdot B \cdot s_{\text{token}} = 2 \cdot \lceil T / B \rceil \cdot B \cdot L \cdot H_{\text{kv}} \cdot d_{\text{head}} \cdot b.$$

Note that the last block may hold fewer than $B$ live KV entries, yet the full block is copied — hence the $\lceil T / B \rceil \cdot B$ term, not $T$.

**Recompute cost** counts the number of tokens whose KV cache must be recomputed via a fresh forward pass:

$$C_{\text{recompute}} = T.$$

## Task

Implement `preemption_costs`:

```python
def preemption_costs(
    seq_len: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    """Return (swap_cost_bytes, recompute_cost_tokens)."""
    ...
```

Given a victim sequence of length `seq_len` and the system parameters listed above, return a 2-tuple:

1. `swap_cost` — total bytes copied in a full round-trip swap (GPU→CPU→GPU).
2. `recompute_cost` — number of tokens that must be re-prefilled.

Both values are plain integers. You may use `math.ceil` or equivalent for the block count. No floating-point intermediate results are permitted — every intermediate value must remain integer.

## Example

```python
swap, recomp = preemption_costs(
    seq_len=100, block_size=16, num_layers=32,
    num_kv_heads=8, head_dim=128, dtype_bytes=2,
)
# N_blocks = ceil(100 / 16) = 7
# s_token  = 32 * 8 * 128 * 2 = 65 536
# swap     = 2 * 7 * 16 * 65 536 = 14 680 064
# recompute = 100
assert swap == 14_680_064
assert recomp == 100
```

## What the gate checks

One gate: `exact_match`. The oracle recomputes both cost figures from the formulas above using `math.ceil` for the block count and checks exact integer equality. Two common mistakes break it:

* Using integer floor division `seq_len // block_size` instead of $\lceil \text{seq\_len} / \text{block\_size} \rceil$ — this undercounts blocks whenever `seq_len` is not a multiple of `block_size`.
* Omitting the factor of $2$ — counting only the GPU→CPU copy, not the full round-trip.

Both bugs silently produce wrong integers for at least some of the random test cases the grader generates.
