## Context

Scaled dot‑product attention for a single query $q \in \mathbb{R}^{d}$ over keys
$K \in \mathbb{R}^{n \times d}$ and values $V \in \mathbb{R}^{n \times d}$ is:

$$
\text{Attention}(q, K, V) = \text{softmax}\!\left(\frac{q K^{\top}}{\sqrt{d}}\right) V,
\qquad \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

In a streaming setting, tokens arrive one by one.  To keep memory bounded,
a _sliding window_ of size $W$ retains only the most recent $W$ keys and values.
However, empirical work (e.g., *Attention Sinks*) has shown that the very first
token (the *sink token*) receives a disproportionate amount of attention, even
when its content is irrelevant.  If the sliding window discards this sink token,
the attention distribution becomes unstable and the model’s output can change
drastically.

## Task

Implement the function `streaming_attention(q, k, v, window_size=4)` that
processes a sequence of tokens step by step.

- `q`, `k`, `v`: NumPy arrays of shape `(T, d)`.  
- `window_size`: number of tokens to retain in the cache (positive integer).  
- Returns a NumPy array of shape `(T, d)`, the output at each step.

**Critical requirement**: The very first token (index 0) must **always** be kept
in the cache as an attention sink.  
This means that at step $t$ the cache contains:

- token $0$, and
- up to $\text{window\_size} - 1$ of the most recent tokens before $t$
  (i.e., indices $[\max(1, t-\text{window\_size}+2), t]$ when $t \ge 1$).

When $t < \text{window\_size}$ the cache holds all tokens $\{0, \dots, t\}$.

**Hint**: The standard sliding window that simply evicts the oldest token when
the cache is full **will** drop the sink token at the first overflow — that is
the bug you must fix.

## Example

Assume $d=1$, $T=3$, $\text{window\_size}=2$.  
Let the data be  

$$
k = \begin{bmatrix}0 \\ 10 \\ 10\end{bmatrix},\qquad
v = \begin{bmatrix}100 \\ 0 \\ 0\end{bmatrix},\qquad
q = \begin{bmatrix}0 \\ 0 \\ 0\end{bmatrix}.
$$

The sink token (index 0) has $k=0$, $v=100$; the others have $k=10$, $v=0$.

At each step, the oracle (sink‑keeping implementation) output is:

- Step 0 (1 token in cache: index 0): output $100$.
- Step 1 (2 tokens: indices 0,1): output $(100+0)/2 = 50$.
- Step 2 ($t=2$, window_size=2).  Since $t$ is not less than window_size,
  the cache must keep token 0 and the most recent token (index 2).  
  Cache: indices 0 and 2, values $[100, 0]$, output $50$.

A **buggy** sliding window that simply evicts the oldest token at step 2
uses cache indices 1 and 2, both with value $0$, so its output is $0$ — a
large deviation.  The oracle output remains close to the full‑attention result
(which would be $(100+0+0)/3 \approx 33.33$) while the buggy output diverges
to $0$.

## What the gate checks

The metric is `max_abs_err`, the maximum absolute difference between your output
and the oracle output (computed with the correct sink‑keeping logic).  The
threshold $1\times10^{-5}$ ensures that only a solution that keeps the sink
token passes.
