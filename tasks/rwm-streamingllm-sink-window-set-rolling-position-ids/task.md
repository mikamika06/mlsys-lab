## Context

Streaming language models often keep a bounded attention state instead of storing the
entire token history. A common strategy is to preserve a small set of attention sink
tokens and a sliding window of recent tokens.

For a sequence of length $T$, the retained token indices are

$$
I = \{0,1,\dots,S-1\} \cup \{T-W,\dots,T-1\},
$$

where $S$ is the number of sink tokens and $W$ is the recent window size. The two
sets are merged and sorted so that each token appears once.

The retained tokens receive rolling position ids. The sink tokens keep their original
positions, while window tokens are shifted so that the final window token has the
latest position value:

$$
p_i =
\begin{cases}
i, & i < S \\
S + i - (T-W), & i \ge T-W
\end{cases}
$$

Attention is computed only over the retained key/value states. For query matrix
$Q$, retained keys $K_I$, and retained values $V_I$, scaled dot-product attention is

$$
\operatorname{Attn}(Q,K_I,V_I)
=
\operatorname{softmax}\left(\frac{QK_I^\top}{\sqrt{d}}\right)V_I .
$$

## Task

Implement `streaming_attention(tokens, q, k, v, S, W)`.

The inputs are NumPy arrays:

- `tokens` is a one-dimensional array containing the token values. Its length is
  $T$.
- `q` has shape $(m,d)$ and contains query vectors.
- `k` has shape $(T,d)$ and contains key vectors.
- `v` has shape $(T,h)$ and contains value vectors.
- `S` is the number of sink tokens.
- `W` is the sliding window size.

Return a tuple:

```python
(retained_indices, position_ids, output)
```

where:

- `retained_indices` is a one-dimensional integer array containing the sink plus
  window token indices.
- `position_ids` contains the rolling position id for each retained index.
- `output` is the scaled dot-product attention result using only the retained
  keys and values.

Use NumPy operations for the attention computation.

## Example

```python
import numpy as np

tokens = np.arange(8)
q = np.array([[1.0, 0.0]])
k = np.eye(8, 2)
v = np.arange(8).reshape(8, 1).astype(float)

idx, pos, out = streaming_attention(tokens, q, k, v, 2, 3)

# idx contains [0, 1, 5, 6, 7]
# pos contains [0, 1, 5, 6, 7]
```

## What the gate checks

The gate builds a NumPy oracle that computes the sink and sliding-window retained
set, the rolling position ids, and the attention output. The returned retained
indices and positions must exactly match the oracle. The attention output must
match the oracle within floating point tolerance.
