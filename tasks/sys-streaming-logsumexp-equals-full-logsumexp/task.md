## Context

The log-sum-exp of a row $x \in \mathbb{R}^D$,

$$
\mathrm{LSE}(x) = \log \sum_{j=1}^{D} e^{x_j} = m + \log \sum_{j=1}^{D} e^{x_j - m}, \qquad m = \max_j x_j,
$$

is the normalizing constant behind softmax and cross-entropy. When $D$ is
too large to hold in memory at once (e.g. attention scores over a very long
context, processed one KV block at a time), it can be computed **online**,
one chunk at a time, without ever materializing the full row. Given chunks
$x^{(1)}, x^{(2)}, \dots$ that partition $x$, maintain a running max $m$ and
running sum $\ell$ (of exponentials rebased to the current running max),
updated after each chunk $x^{(k)}$ with local max $m_k = \max_j x^{(k)}_j$:

$$
m_{\text{new}} = \max(m, m_k), \qquad
\ell_{\text{new}} = \ell \cdot e^{\,m - m_{\text{new}}} + \sum_j e^{\,x^{(k)}_j - m_{\text{new}}},
$$

initialized with $m = -\infty$, $\ell = 0$. After the last chunk, the exact
row LSE is $m + \log \ell$ — identical to the value you would get by
computing the LSE over the full row directly, but without ever holding more
than one chunk in memory at a time.

## Task

Implement `streaming_logsumexp(chunks)`.

`chunks` is a list of 2D `float` arrays, all with the same number of rows
$N$, that together tile a full $(N, D)$ score matrix along the column axis
(`np.concatenate(chunks, axis=1)` would reconstruct it, though you must not
do that). Process the chunks **one at a time**, in order, maintaining a
running per-row max and running per-row sum as described above (do not
concatenate the chunks and take a single full log-sum-exp — the whole point
is that you never hold more than one chunk's worth of columns in memory at
once). Return the per-row LSE as a `(N,)` `float64` NumPy array.

## Example

```python
import numpy as np

full = np.array([[1.0, 2.0, 3.0, 0.5]])
chunks = [full[:, :2], full[:, 2:]]   # two chunks of width 2

result = streaming_logsumexp(chunks)
# same value as: m = full.max(1); m + np.log(np.exp(full - m[:, None]).sum(1))
```

## What the gate checks

The gate builds random score matrices, splits them into chunks of varying
widths (in one case 2 wide chunks, in another 5 narrower ones), and:

- **`max_abs_err`**: compares your per-row result against a directly
  computed full-row reference, `m + log(sum(exp(full - m)))` on the
  reconstructed matrix, with $\max_i |y_i - \hat{y}_i| \le 10^{-6}$.
- **`streamed`**: temporarily wraps `np.exp` during your call and flags a
  violation if it is ever invoked on an array wider (more columns) than the
  widest individual chunk you were given — i.e. if you secretly
  concatenated the chunks and exponentiated the full row at once instead of
  processing chunks one at a time. Must be `1.0` (no violation) on every
  case.

Both gates must hold.
