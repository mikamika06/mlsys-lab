## Context

Transformer attention computes a probability distribution over tokens. Given an attention
logit matrix $L \in \mathbb{R}^{n \times n}$, the attention matrix is produced by
applying a row-wise softmax:

$$
A_{ij} = \frac{\exp(L_{ij})}{\sum_{m=1}^{n}\exp(L_{im})}.
$$

The sink phenomenon describes the observation that early tokens can absorb a large
fraction of the total attention probability mass. The total mass received by token
$j$ is the sum of its attention column:

$$
c_j = \sum_{i=1}^{n} A_{ij}.
$$

For a chosen number of initial tokens $k$, the sink probability mass fraction is

$$
s_k = \frac{\sum_{j=1}^{k} c_j}{\sum_{j=1}^{n} c_j}.
$$

Since every attention row sums to $1$, the denominator is normally $n$, but it is
computed from the attention matrix to keep the measurement definition explicit.

## Task

Implement `attention_sink_mass(logits, k)`:

```python
def attention_sink_mass(logits: list[list[float]], k: int) -> float:
    ...
```

The function receives a square list of lists of floats of attention logits with shape
$(n,n)$ and an integer $k$. It must:

1. Compute the row-wise softmax attention matrix in numerically stable form.
2. Compute the total attention probability mass assigned to the first $k$ tokens.
3. Return the sink mass fraction as a Python `float`.

Do not use external machine learning libraries.

## Example

```python

logits = [
    [4.0, 1.0, 0.0],
    [3.0, 2.0, 0.0],
    [2.0, 1.0, 0.0],
]

mass = attention_sink_mass(logits, 1)
# mass is the fraction of attention probability assigned to token 0
```

## What the gate checks

The gate builds several attention logit matrices and computes the reference sink
mass by constructing the stable Python softmax attention matrix. The returned value
is compared with the oracle result using relative error:

$$
\mathrm{rel\_err} =
\frac{|x-\hat{x}|}{|x|+10^{-12}}.
$$

The gate passes when $\mathrm{rel\_err} \leq 10^{-6}$.
