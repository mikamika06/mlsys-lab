## Context

Long-context attention systems often reduce memory by keeping a small set of sink tokens together with a sliding window of recent tokens. For a stream of tokens, the live cache at step $t$ contains the union of sink indices and recent indices:

$$
C_t = \{0,1,\dots,k-1\} \cup \{\max(0,t-w+1),\dots,t\}.
$$

Given a query $q_t$, keys $K$ and values $V$, attention over the kept cache is

$$
\mathrm{Attn}(q_t,K_{C_t},V_{C_t})
=
\sum_{i \in C_t}
\frac{\exp(q_t^\top K_i / \sqrt{d})}
{\sum_{j \in C_t}\exp(q_t^\top K_j / \sqrt{d})}
V_i .
$$

The eviction policy does not change the attention computation itself. It only changes which key/value pairs remain available. The implementation must replay the stream while ensuring the cache never grows beyond the sink tokens plus the recent window.

## Task

Implement `sink_attention_stream(Q, K, V, k, w)`:

```python
def sink_attention_stream(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    k: int,
    w: int,
) -> tuple[np.ndarray, list[int]]:
    ...
```

The inputs are:

- `Q` with shape $(n,d)$ containing one query per stream step.
- `K` with shape $(n,d)$ containing streamed keys.
- `V` with shape $(n,m)$ containing streamed values.
- `k` sink token count.
- `w` recent sliding-window size.

For every step $t$, insert the current token into the live cache, evict tokens that are neither sinks nor in the recent window, and compute attention using only the kept indices. Return:

- an array of shape $(n,m)$ containing the attention output for every step.
- a list containing the live cache length after every step.

Use NumPy operations for the attention calculation. The returned outputs must use `float64` arithmetic.

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.]])
V = np.array([[10., 0.], [0., 20.]])

out, lengths = sink_attention_stream(Q, K, V, k=1, w=1)

# The first step attends to token 0.
# The second step keeps sink token 0 and recent token 1.
# lengths == [1, 2]
```

## What the gate checks

The gate replays several streams and computes a NumPy reference implementation that forms the kept index set at every step. The returned attention outputs must satisfy

$$
\max_i |y_i-\hat{y}_i| \le 10^{-5}.
$$

The cache lengths must also satisfy the eviction rule and never exceed $k+w$ tokens. Implementations that only keep the recent window and drop sinks will fail because their attention results differ from the oracle.
