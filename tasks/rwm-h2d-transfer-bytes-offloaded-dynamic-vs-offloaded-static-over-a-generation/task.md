## Context

During autoregressive generation in a transformer, the key-value (KV) cache grows by one entry per decode step. When GPU memory is scarce the KV cache is *offloaded* to host (CPU) RAM, and entries must be transferred host-to-device (H2D) before each forward pass. Two offloading strategies are common.

**Dynamic (full re-staging).** The entire KV cache lives on the host. Before every decode step $t$ the full cache — containing $L + t$ token entries — is transferred H2D. The total H2D bytes over $T$ decode steps are

$$B_{\text{dyn}} = b \sum_{t=0}^{T-1}(L + t) = b\!\left(TL + \frac{T(T-1)}{2}\right),$$

where $b$ is the KV bytes per token (across all layers and heads) and $L$ is the prompt length.

**Static (fixed slab).** A device-side slab of `max_len` entries is allocated once. After prefill, the $L$ existing entries are transferred H2D. At each subsequent decode step only the 1 newly computed KV entry is transferred:

$$B_{\text{st}} = b(L + T).$$

The static strategy costs $O(L + T)$ bandwidth versus $O(T(L + T))$ for dynamic — a factor of roughly $T/2$ — at the expense of reserving a contiguous device slab.

A well-known production offloading library implements exactly this comparison to decide which strategy to use for a given prompt budget.

## Task

Implement

```python
def h2d_transfer_bytes(L: int, layer_bytes: int, T: int, max_len: int) -> tuple[int, int]:
    """Return (dynamic_bytes, static_bytes) — total H2D transfer in bytes
    over T decode steps for the dynamic and static offloading strategies.

    Parameters
    ----------
    L          : prompt length (number of prefill tokens).
    layer_bytes: KV bytes per token across all layers.
    T          : number of decode (generation) steps.
    max_len    : maximum context length the slab can hold.

    Raises ValueError if L + T > max_len.
    """
```

Use integer arithmetic throughout. Do **not** import NumPy.

## Example

```python
h2d_transfer_bytes(L=4, layer_bytes=16, T=3, max_len=8)
# Dynamic: 16*(4 + 5 + 6) = 16*15 = 240
# Static:  16*(4 + 3)     = 16*7  = 112
# Returns (240, 112)
```

```python
h2d_transfer_bytes(L=100, layer_bytes=256, T=50, max_len=100)
# Raises ValueError  (100 + 50 > 100)
```

## What the gate checks

A single `exact_match` gate. The grader runs its own step-by-step loop oracle for several parameter tuples (including an edge case with $T = 0$ and a `ValueError` case) and compares the returned pair against the oracle's result. Both integers in the tuple must match exactly; any wrong value, wrong type, or missing `ValueError` causes the gate to fail.
