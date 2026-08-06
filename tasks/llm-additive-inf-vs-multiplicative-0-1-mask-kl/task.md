## Context

In transformer‑style attention, a *mask* is applied to the raw logits before the softmax so that certain positions are ignored.  
The standard way is **additive** masking:

$$\tilde{l}_{ij} = l_{ij} + \begin{cases}
0 & \text{if } m_{ij}=1\\
-\infty & \text{if } m_{ij}=0
\end{cases},$$

followed by the softmax

$$p_{ij} = \frac{\exp(\tilde{l}_{ij})}{\sum_k \exp(\tilde{l}_{ik})}.$$

An alternative, but incorrect, approach is to **multiply** the logits or probabilities by a 0/1 mask after the softmax.  
This distorts the distribution because it no longer sums to one and changes relative scores.

The task below asks you to implement the correct additive‑masking routine and verify its correctness using the mean Kullback–Leibler divergence between your output and a reference implementation.

## Task

Implement `masked_softmax(logits, mask)`:

```python
def masked_softmax(logits: list[list[float]], mask: list[list[int]]) -> list[list[float]]:
    ...
```

* `logits` is a 2‑D list of shape `(batch, seq_len)` containing raw attention scores.
* `mask` has the same shape and contains boolean or integer values (`1`/`0`) indicating which positions are allowed to attend.
* The function must return a probability matrix of the same shape where masked entries are exactly zero and each row sums to one.
* Use only vectorised Python operations; no explicit Python loops.

## Example

```python
logits = [[1.0, 2.0, -1.0],
                   [0.5, 0.0, 3.0]]
mask   = [[1, 0, 1],
                   [1, 1, 0]]

probs = masked_softmax(logits, mask)
print(probs)  # [[0.8807970779778823, 0.0, 0.11920292202211755], [0.6224593312018546, 0.37754066879814546, 0.0]]
```

## What the gate checks

The grader computes a reference distribution using additive `-inf` masking and then evaluates the mean Kullback–Leibler divergence between that reference and your output:

$$\text{mean\_kl} = \frac{1}{B}\sum_{b=1}^{B}
   \operatorname{KL}\!\bigl(p^{\text{ref}}_b\,||\,p^{\text{cand}}_b\bigr).$$

The gate requires `mean_kl <= 1e-6`.  
A correct implementation will produce a negligible divergence; an implementation that zeroes logits or probabilities multiplicatively will yield a large divergence and fail the gate.
