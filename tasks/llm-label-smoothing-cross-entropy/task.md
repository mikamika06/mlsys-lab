## Context

The cross‑entropy loss for a single example with logits $\mathbf{z}\in\mathbb R^K$ and target class $y\in\{0,\dots,K-1\}$ is  

$$
\ell(\mathbf{z}, y) = - \log \frac{\exp(z_y)}{\sum_{k=1}^{K} \exp(z_k)} .
$$

Label‑smoothing replaces the one‑hot target $\delta_y$ with a softened distribution  

$$
q_k = (1-\varepsilon)\,\delta_{y,k} + \frac{\varepsilon}{K},
$$

where $0\le\varepsilon<1$.  
The smoothed loss becomes  

$$
\ell_{\text{smooth}}(\mathbf{z}, y) = - \sum_{k=1}^{K} q_k\,\log p_k,
$$

with $p_k=\frac{\exp(z_k)}{\sum_j \exp(z_j)}$ the softmax probabilities.

## Task

Implement a function that computes the average label‑smoothed cross‑entropy over a batch:

```python
def label_smoothed_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                                 eps: float = 0.1) -> float:
    ...
```

* `logits` – shape $(N,K)$ NumPy array of raw logits for $N$ examples and $K$ classes.
* `targets` – shape $(N,\,)$ integer class indices in $\{0,\dots,K-1\}$.
* `eps` – smoothing factor $\varepsilon$.

The function must return a **float64** scalar equal to the mean loss over the batch.  
Use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
logits = np.array([[0.2, 0.8], [1.5, -0.5]])
targets = np.array([1, 0])
loss = label_smoothed_cross_entropy(logits, targets, eps=0.1)
print(loss)   # ≈ 0.693147
```

## What the gate checks

The grader evaluates the candidate against a NumPy reference implementation on several random batches.  
It reports the maximum absolute error between the candidate’s scalar loss and the reference.  
The solution must satisfy  

$$\max_{\text{cases}}\;|\,\hat{\ell} - \ell_{\text{ref}}\,|\;\le 10^{-6}.$$

Any deviation larger than this threshold causes the gate to fail.
