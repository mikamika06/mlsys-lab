## Context

Long-context language model serving often uses a retention policy to decide which
tokens remain in the attention cache when the cache budget is smaller than the
context length.

StreamingLLM keeps a fixed set of attention sink tokens at the beginning of the
sequence and a recent window of tokens. With a window size $w$ and sink count
$s$, the retained set at time $t$ is

$$
R_{\mathrm{stream}} = \{0,\dots,s-1\} \cup \{t-w+1,\dots,t\}.
$$

H2O-style retention uses heavy-hitter tokens. It tracks accumulated attention
mass for tokens and keeps the tokens with the largest historical importance. If
the cache budget is $b$, the retained set is

$$
R_{\mathrm{h2o}} = \operatorname{TopB}\left(\sum_{q} A_{q,:}, b\right),
$$

where $A$ is the attention matrix accumulated over queries and
$\operatorname{TopB}$ selects the indices with the largest values.

The attention mass of a retained set $R$ is

$$
M(R) = \sum_{i \in R} h_i,
$$

where $h_i$ is the accumulated attention score of token $i$.

## Task

Implement `compare_retention(attention, window_size, budget, needle_index)`.

The input `attention` is a list of lists of floats where rows are queries and columns
are tokens. Column sums represent accumulated attention mass. The function must
return a dictionary:

```python
{
    "streaming_retained": [...],
    "h2o_retained": [...],
    "streaming_keeps_needle": bool,
    "h2o_keeps_needle": bool,
    "streaming_mass": float,
    "h2o_mass": float,
}
```

Use the following rules:

- StreamingLLM uses `2` sink tokens, indices `0` and `1`, plus the most recent
  `window_size` tokens.
- H2O keeps exactly `budget` tokens with the highest accumulated attention mass.
- When selecting tied values, prefer the smaller token index.
- Retained indices must be sorted in ascending order.
- Attention masses must be computed from the accumulated column sums.

## Example

```python

attention = [
    [0.2, 0.1, 0.1, 0.1, 0.5],
    [0.1, 0.1, 0.1, 0.6, 0.1],
]

result = compare_retention(attention, 2, 3, 2)

# The returned dictionary contains the retained indices and whether token 2
# survives each policy.
```

## What the gate checks

The gate generates several attention streams containing a high-attention token
outside the recent window. It computes the StreamingLLM and H2O results with a
Python oracle using the definitions above.

The `exact_match` gate requires every returned field to match the oracle,
including retained token indices, needle retention flags, and retained attention
mass values.
