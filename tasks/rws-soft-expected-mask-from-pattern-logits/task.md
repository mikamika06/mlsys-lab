## Context

In many machine‑learning pipelines a set of *patterns* is weighted by the probability distribution produced from a set of logits.  
Let $L \in \mathbb{R}^{B\times P}$ be a batch of logits, where $B$ is the number of examples and $P$ the number of patterns.  The softmax over each row gives a probability vector

$$
s_{b,p} = \frac{\exp(L_{b,p})}{\sum_{q=1}^{P}\exp(L_{b,q})},
$$

which we denote by $S$.  
Let $M \in \mathbb{R}^{P\times D}$ be the pattern matrix, each row being a *mask* of dimension $D$.  The **soft expected mask** is then defined as

$$
E = S\, M,
$$

i.e. for every example we take the weighted sum of all patterns according to their softmax probabilities.

This operation appears in attention‑like modules and in probabilistic mixture models where a continuous output is obtained by averaging discrete prototypes.

## Task

Implement the function `soft_expected_mask`:

```python
def soft_expected_mask(logits: list[list[float]], patterns: list[list[float]]) -> list[list[float]]:
    ...
```

* `logits` – a 2‑D list of shape `(B, P)` containing arbitrary real numbers.
* `patterns` – a 2‑D list of shape `(P, D)` containing the pattern masks.

The function must return a list of shape `(B, D)` with dtype `float64`.  
Use only vectorised Python operations; no explicit Python loops are allowed.

## Example

```python

logits   = [[0.0, 1.0], [2.0, -1.0]]
patterns = [[1.0, 0.0],
                     [0.0, 1.0]]

mask = soft_expected_mask(logits, patterns)
print(mask)  # [[0.2689414213699951, 0.7310585786300049], [0.9525741268224334, 0.04742587317756679]]
```

The first row is the softmax of `[0,1]` times the identity pattern matrix; the second row uses the softmax of `[2,-1]`.

## What the gate checks

The grader computes a Python reference `E_ref = softmax(logits) @ patterns`.  
Your implementation must produce an array whose global relative L2 error satisfies

$$
\mathrm{rel\_err} = \frac{\lVert E_{\text{cand}} - E_{\text{ref}}\rVert}{\lVert E_{\text{ref}}\rVert}
\le 10^{-6}.
$$

The gate will fail if the relative error exceeds this threshold.
