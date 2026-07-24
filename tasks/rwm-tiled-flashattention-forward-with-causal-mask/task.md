## Context

Attention computes a weighted combination of value vectors using query-key similarities. Given query matrix $Q \in \mathbb{R}^{n \times d}$, key matrix $K \in \mathbb{R}^{n \times d}$, and value matrix $V \in \mathbb{R}^{n \times d}$, the standard forward pass is

$$
S = \frac{QK^\top}{\sqrt{d}},
$$

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}},
$$

$$
O = PV.
$$

For causal language models, token $i$ cannot attend to future tokens. The causal mask sets

$$
S_{ij} = -\infty \quad \text{when } j > i .
$$

A production FlashAttention kernel avoids materializing the full $n \times n$ attention matrix. It processes blocks of keys and values while maintaining a running softmax normalization. For a query row block, each key block updates the running maximum $m$ and normalization factor $l$:

$$
m_{\text{new}} = \max(m, m_{\text{block}})
$$

and

$$
l_{\text{new}} =
e^{m-m_{\text{new}}}l +
\sum_j e^{s_j-m_{\text{new}}}.
$$

The output accumulator is rescaled using the same normalization update before adding the contribution from the current block.

## Task

Implement `flash_attention_forward(Q, K, V, block_size=2)`.

The function receives three NumPy arrays of shape $(n, d)$ with dtype `float64` and returns a NumPy array of shape $(n, d)$ containing causal scaled dot-product attention.

Requirements:

- Implement the tiled algorithm by iterating over query rows and key/value blocks.
- Apply the causal mask so position $i$ never uses keys $j > i$.
- Maintain running softmax statistics instead of constructing the complete attention matrix.
- Use only NumPy operations inside the kernel. The function must return `float64`.

The expected function signature is:

```python
def flash_attention_forward(Q, K, V, block_size=2):
    ...
```

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.]])
V = np.array([[1., 2.], [3., 4.], [5., 6.]])

O = flash_attention_forward(Q, K, V, block_size=2)
```

The result matches the causal softmax attention computed by applying the mask before softmax.

## What the gate checks

The gate computes a NumPy oracle that performs full causal attention with the complete score matrix. The submission is evaluated on several deterministic inputs and its output is compared with the oracle.

The reported metric is the maximum absolute error

$$
\max_{i,j}|O_{ij}^{\mathrm{submission}}-O_{ij}^{\mathrm{oracle}}|.
$$

The value must be at most $10^{-6}$. Incorrect masking, unstable normalization updates, or implementations that ignore tiled processing will fail.
