## Context

In a language model the final linear layer (the *head*) maps hidden states to logits over the vocabulary.  
If $H \in \mathbb{R}^{B\times T\times d}$ denotes the hidden activations for a batch of size $B$ and sequence length $T$, the weight matrix $W_{\text{out}}\in\mathbb{R}^{V\times d}$ (where $V$ is the vocabulary size) and bias $b \in \mathbb{R}^V$, the logits are

$$
L_{b,t,v}= H_{b,t,:}\, \bigl(W_{\text{out}}^{\top}\bigr)_{v,:}+ b_v .
$$

The operation can be expressed as a batched matrix multiplication followed by broadcasting of the bias.

## Task

Implement `lm_head_projection(hidden_states, weight, bias)`:

```python
def lm_head_projection(hidden_states: np.ndarray,
                       weight: np.ndarray,
                       bias: np.ndarray) -> np.ndarray:
    ...
```

The function receives:

- `hidden_states`: a 3‑D NumPy array of shape `(batch, seq_len, hidden_dim)`
- `weight`: a 2‑D array of shape `(vocab_size, hidden_dim)`
- `bias`: a 1‑D array of shape `(vocab_size,)`

It must return the logits as a float64 array of shape `(batch, seq_len, vocab_size)`.  
Only NumPy operations are allowed; no explicit Python loops.

## Example

```python
import numpy as np

hidden = np.array([[[0., 1.], [2., 3.]]])          # shape (1,2,2)
weight = np.array([[1., 0.], [0., 1.]])            # shape (2,2)
bias   = np.array([0., 1.])                       # shape (2,)

logits = lm_head_projection(hidden, weight, bias)
print(logits)
# [[[0. 1.]
#   [2. 4.]]]
```

## What the gate checks

The grader computes a reference implementation with NumPy and compares your output using the metric `max_abs_err`.  
Your solution must satisfy

$$
\mathrm{max\_abs\_err} \le 10^{-5}.
$$

A correct implementation uses `np.matmul` (or the `@` operator) followed by bias addition. Any deviation, such as omitting the bias or transposing incorrectly, will produce a larger error and fail the gate.
