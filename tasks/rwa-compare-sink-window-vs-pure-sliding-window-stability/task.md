## Context

Long-context attention systems often avoid storing all previous tokens by keeping a small
attention memory. A common strategy is sink-plus-window attention: preserve a few early
tokens called sinks and keep a recent sliding window.

For a query $q_t$, keys $K$ and values $V$, attention is computed as

$$
p_t = \mathrm{softmax}\left(\frac{q_t K^\top}{\sqrt{d}}\right),
$$

and the output is

$$
o_t = p_t V.
$$

The entropy of the attention distribution is

$$
H(p_t) = -\sum_i p_{t,i}\log(p_{t,i}).
$$

A full-attention reference uses every token from the beginning of the stream. Sink-plus-window
keeps the first $s$ tokens and the newest $w$ tokens. Pure sliding-window keeps only the
newest $w$ tokens. When a stream becomes long, removing the sink tokens can change the
attention distribution and make entropy behavior diverge from full attention.

## Task

Implement `compare_sink_window(Q, K, V, sink_size, window_size)`.

The inputs are NumPy arrays:

- `Q` has shape $(T, d)$ and contains one query per time step.
- `K` has shape $(T, d)$ and contains stream keys.
- `V` has shape $(T, d_v)$ and contains stream values.

For every time step $t$, compare the two memory policies:

1. Sink-plus-window: attend over token indices $[0, s)$ and the most recent
   `window_size` tokens ending at $t`.
2. Pure sliding-window: attend only over the most recent `window_size` tokens ending
   at $t$.

Return a dictionary with these four NumPy arrays:

```python
{
    "sink_outputs": ...,
    "pure_outputs": ...,
    "sink_entropy": ...,
    "pure_entropy": ...
}
```

The output arrays must use `float64`. Compute attention probabilities with numerically
stable softmax.

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
K = Q.copy()
V = np.array([[1., 0.], [0., 1.], [2., 2.]])

result = compare_sink_window(Q, K, V, sink_size=1, window_size=2)

print(result["sink_outputs"].shape)
# (3, 2)
```

## What the gate checks

The gate builds a NumPy oracle that computes full attention, sink-plus-window attention,
pure sliding-window attention, and attention entropy in `float64`.

The returned values are flattened together and compared against the oracle using relative
error:

$$
\mathrm{rel\_err} =
\frac{\lVert x_{\mathrm{student}} - x_{\mathrm{oracle}}\rVert_2}
{\lVert x_{\mathrm{oracle}}\rVert_2 + 10^{-12}} .
$$

The result must satisfy $\mathrm{rel\_err} \le 10^{-4}$. Implementations that only
return the recent window or omit the entropy computation will not match the oracle.
