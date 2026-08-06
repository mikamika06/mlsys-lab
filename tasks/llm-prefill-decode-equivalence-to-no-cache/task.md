## Context

In an autoregressive language model the hidden state produced at step $t$ is used to compute the logits for token $t+1$. When decoding we usually keep a key/value cache so that each new token can be processed in $O(1)$ time. If the cache is implemented correctly, running the model once with a full prefill of all tokens and then decoding step‑by‑step should produce exactly the same sequence of hidden states as recomputing the whole prefix from scratch at every step.

The following task demonstrates this equivalence for a toy recurrent neural network. The network has parameters $W_{ih}, W_{hh}\in\mathbb{R}^{d\times d}$ and bias $b\in\mathbb{R}^d$ and updates its hidden state with

$$h_t = \tanh(W_{ih}\,x_t + W_{hh}\,h_{t-1} + b).$$

For a sequence of embeddings $X=(x_0,\dots,x_{n-1})$ we can compute the hidden states in two ways:

* **No‑cache** – for each prefix length $\ell$ recompute all hidden states from scratch.
* **Cache** – maintain the previous hidden state and update it once per token.

Both procedures should yield identical arrays of shape $(n,d)$.

## Task

Implement `prefill_decode_equiv(inputs: list[float]) -> Tuple[list[float], list[float]]`:

```python
def prefill_decode_equiv(inputs: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

The function receives a 2‑D list `inputs` of shape `(seq_len, d)` and returns a tuple `(no_cache, cache)` where each element is an array of shape `(seq_len, d)`. The first array contains the hidden states obtained by the no‑cache strategy; the second contains those obtained with a key/value cache. Both must be computed using only Python operations and `float64` precision.

## Example

```python
from solution_ref import prefill_decode_equiv   # or your own implementation

rng = random.Random(42)
inputs = rng.randn(5, 16).astype(float)

no_cache, cache = prefill_decode_equiv(inputs)
print(no_cache.shape)  # (5, 16)
print(cache.shape)     # (5, 16)
```

The two arrays should be numerically identical up to machine precision.

## What the gate checks

The grader computes a reference implementation and compares your output with it. The maximum absolute difference between corresponding elements of the two returned arrays must satisfy

$$\max_{i,j}\bigl|\text{your}_{ij}-\text{ref}_{ij}\bigr| \le 10^{-5}.$$

If this condition is violated the solution fails the gate.
