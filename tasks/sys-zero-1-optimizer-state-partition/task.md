## Context

Training large neural networks requires optimizer states (momentum, variance) that can
double or triple memory consumption.  **ZeRO Stage 1** (Zero Redundancy Optimizer,
partition stage 1) shards these optimizer states across $R$ data-parallel ranks while
each rank retains a full copy of the model parameters.

For the Adam optimizer, per-parameter element $i$ at step $t$:

$$m_i^{(t)} = \beta_1 \, m_i^{(t-1)} + (1 - \beta_1)\, g_i^{(t)}$$

$$v_i^{(t)} = \beta_2 \, v_i^{(t-1)} + (1 - \beta_2)\, (g_i^{(t)})^2$$

$$\hat{m}_i^{(t)} = \frac{m_i^{(t)}}{1 - \beta_1^{\,t}}, \qquad
  \hat{v}_i^{(t)} = \frac{v_i^{(t)}}{1 - \beta_2^{\,t}}$$

$$\theta_i^{(t)} = \theta_i^{(t-1)} - \eta \,
  \frac{\hat{m}_i^{(t)}}{\sqrt{\hat{v}_i^{(t)}} + \epsilon}$$

Because every element update depends only on that element's own gradient history,
the $N$ parameters can be partitioned into $R$ disjoint shards without affecting
the result.  Rank $r$ owns indices $S_r \subset \{0, \ldots, N-1\}$, stores only
the corresponding rows of $m$ and $v$, and applies the Adam step to only those
elements.  After each step an implicit all-gather reconstructs the full parameter
vector.

The contiguous partition assigns the first $\lceil N / R \rceil$ elements to rank 0,
the next block to rank 1, and so on, with the last rank absorbing any remainder:

$$|S_r| = \begin{cases}
  \lceil N/R \rceil & \text{if } r < N \bmod R, \\
  \lfloor N/R \rfloor & \text{otherwise.}
\end{cases}$$

## Task

Implement `zero_one_adam`:

```python
def zero_one_adam(params, grads, num_ranks, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """ZeRO-1 Adam: partition optimizer states across num_ranks.

    params:     1-D array, shape (N,)   — initial parameters
    grads:      2-D array, shape (T, N) — one gradient vector per step
    num_ranks:  int — number of ranks to shard optimizer states across
    lr, beta1, beta2, eps: Adam hyper-parameters

    Returns:    1-D array, shape (N,) — updated parameters after T steps
    """
```

Your implementation **must** maintain separate per-rank optimizer state arrays
(momentum `m` and variance `v`) and update only each rank's shard at every step.
You may not simply run unsharded Adam over the full parameter vector.

## Example

```python
import numpy as np
rng = np.random.RandomState(0)
N, T, R = 12, 4, 3
params = rng.randn(N)
grads  = rng.randn(T, N)

result = zero_one_adam(params, grads, num_ranks=R)
# result.shape == (12,)
```

Rank 0 owns indices 0–3, rank 1 owns 4–7, rank 2 owns 8–11.
After 4 steps, the returned vector matches a standard Adam run element-for-element.

## What the gate checks

The gate computes a **standard unsharded Adam** reference using the same
hyper-parameters on the same `(params, grads)` input, then reports

$$\texttt{max\_abs\_err} = \max_i \bigl| \texttt{result}_i - \texttt{reference}_i \bigr|$$

A correct ZeRO-1 partition produces bit-identical element-wise updates, so
$\texttt{max\_abs\_err}$ must be below $10^{-6}$.  Returning zeros, the wrong
shape, or any other incorrect result will exceed this threshold and fail the gate.
