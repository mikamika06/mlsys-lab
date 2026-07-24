## Context

How many sequences a server can run at once is usually capped by KV-cache
VRAM, not compute. For a model with `num_layers` layers and
`num_kv_heads` KV heads of dimension `head_dim`, storing **both** K and V
for one token costs:

$$
\text{bytes\_per\_token} = 2 \cdot \text{num\_layers} \cdot \text{num\_kv\_heads} \cdot \text{head\_dim} \cdot \text{bytes\_per\_elem}
$$

where `bytes_per_elem` is `1` for fp8 and `2` for fp16 — halving the
per-element cost roughly doubles how many sequences fit. If every
sequence is assumed to run up to `seq_len` tokens, one sequence's full KV
footprint is `bytes_per_token * seq_len`, and the number that fit in a
`vram_budget_bytes` VRAM budget is:

$$
\text{max\_concurrent\_sequences} = \left\lfloor \frac{\text{vram\_budget\_bytes}}{\text{bytes\_per\_token} \cdot \text{seq\_len}} \right\rfloor
$$

## Task

Implement `kv_capacity`:

```python
def kv_capacity(config: dict, vram_budget_bytes: int) -> dict:
    ...
```

- `config`: a dict with keys `num_layers`, `num_kv_heads`, `head_dim`,
  `seq_len` (all positive ints).
- `vram_budget_bytes`: positive int, the VRAM budget available for the KV
  cache.

Return a dict with four keys:

- `"bytes_per_token_fp8"`, `"bytes_per_token_fp16"` — per the formula
  above, with `bytes_per_elem` of `1` and `2` respectively.
- `"max_concurrent_fp8"`, `"max_concurrent_fp16"` — the corresponding
  `max_concurrent_sequences`, per the floor-division formula above.

## Example

```python
config = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "seq_len": 4096}
kv_capacity(config, vram_budget_bytes=40 * 1024**3)
# bytes_per_token_fp8  = 2*32*8*128*1 = 65536
# bytes_per_token_fp16 = 2*32*8*128*2 = 131072
# max_concurrent_fp8  = floor(40*1024**3 / (65536*4096))
# max_concurrent_fp16 = floor(40*1024**3 / (131072*4096))
# (fp8 fits roughly twice as many concurrent sequences as fp16)
```

## What the gate checks

The grader builds several `(config, vram_budget_bytes)` scenarios from a
seeded NumPy generator (varying layer counts, KV head counts, head dims,
sequence lengths, and VRAM budgets — including budgets that don't divide
evenly, to exercise the floor) and computes all four reference values
independently with exact integer arithmetic from the formulas above,
never calling your function or hardcoding an expected value.

`exact_match` is the fraction of scenarios where all four of your
returned values equal the oracle's exactly, and the gate requires `1.0`.
Using `num_layers` or `num_kv_heads` alone without both, forgetting the
factor of 2 for K+V, swapping the fp8/fp16 byte-per-element multiplier,
or rounding instead of flooring the concurrency count will all produce a
mismatch.
