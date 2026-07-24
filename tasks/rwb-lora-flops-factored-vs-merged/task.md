## Context

Low-Rank Adaptation (LoRA) applies an update to a weight matrix without storing a full dense update. For an input vector $x \in \mathbb{R}^{d}$, the factored update uses two smaller matrices:

$$
xAB = (xA)B,
$$

where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ with rank $r \ll d$.

A matrix multiplication between shapes $m \times k$ and $k \times n$ requires $2mkn$ floating point operations (FLOPs). Therefore, applying the LoRA factors to one token costs:

$$
F_{\mathrm{factored}} = 2dr + 2rd = 4dr .
$$

A production system may merge the LoRA update into a dense matrix before inference:

$$
W_{\mathrm{merged}} = AB .
$$

The merge itself costs:

$$
F_{\mathrm{merge}} = 2d^2r ,
$$

and applying the merged matrix to one token costs:

$$
F_{\mathrm{merged}} = 2d^2 .
$$

For a sequence of length $s$, the two strategies have total costs:

$$
T_{\mathrm{factored}}(s) = s(4dr),
$$

$$
T_{\mathrm{merged}}(s) = 2d^2r + s(2d^2).
$$

The break-even sequence length is the smallest integer $s$ where merging is no more expensive:

$$
s \geq \frac{2d^2r}{4dr - 2d^2}.
$$

The denominator is only positive when the factored representation has a lower per-token cost. If it is not positive, merging cannot amortize its one-time cost.

## Task

Implement `lora_break_even(d, r, max_sequence_length)`:

```python
def lora_break_even(d: int, r: int, max_sequence_length: int) -> dict:
    ...
```

Return a dictionary with exactly these keys:

- `factored_flops_per_token`: the FLOPs for applying the two LoRA factors to one token.
- `merged_flops_per_token`: the FLOPs for applying the merged dense matrix to one token.
- `merge_flops`: the one-time FLOPs required to merge the LoRA matrices.
- `break_even_sequence_length`: the smallest integer sequence length where the merged approach has total FLOPs less than or equal to the factored approach, or `None` if no such length exists up to `max_sequence_length`.

All returned FLOP values must be integers. The break-even calculation must use the total FLOP equations above rather than an approximation.

## Example

```python
result = lora_break_even(4096, 8, 1000)

# {
#   "factored_flops_per_token": 131072,
#   "merged_flops_per_token": 33554432,
#   "merge_flops": 268435456,
#   "break_even_sequence_length": None
# }
```

## What the gate checks

The gate recomputes the FLOP formulas from an independent reference implementation for several values of $d$, $r$, and maximum sequence length.

The returned dictionary must exactly match the oracle output. The gate checks the integer FLOP formulas and the amortization threshold calculation.
