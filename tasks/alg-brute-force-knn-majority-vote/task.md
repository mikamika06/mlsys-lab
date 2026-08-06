## Context

The $k$‑Nearest Neighbours ($k$NN) algorithm classifies a query point by looking at the labels of its $k$ closest training points in Euclidean space.  
Let $\mathcal{X}\in\mathbb{R}^{n\times d}$ be the matrix of $n$ training samples and $\mathbf{y}\in\mathbb{Z}^n$ their integer class labels. For a test point $\mathbf{x}\in\mathbb{R}^d$, we compute its Euclidean distance to every training sample, pick the $k$ smallest distances, collect the corresponding labels, and predict the most frequent one.  
When several classes tie for the majority, the algorithm must break ties deterministically by choosing the **smallest** label.

## Task

Implement a function with the following signature:

```python
def knn_majority_vote(Xtr: list[list[float]],
                      ytr: list[int],
                      Xte: list[list[float]],
                      k: int) -> list[int]:
    ...
```

* `Xtr` – training data, shape $(n_{\text{train}}, d)$, dtype float64.  
* `ytr` – integer labels for the training data, shape $(n_{\text{train}},)$, dtype int64.  
* `Xte` – test data to classify, shape $(n_{\text{test}}, d)$, dtype float64.  
* `k` – number of neighbours to consider (positive integer).  

The function must return a list of shape $(n_{\text{test}},)$ containing the predicted labels for each test point. The implementation should be **brute‑force**: compute all pairwise distances explicitly; no external libraries such as scikit‑learn are allowed.

## Example

```python

Xtr = [[0, 0], [1, 0], [0, 2]]
ytr = [0, 1, 1]
Xte = [[0.5, 0.5], [2, 2]]

preds = knn_majority_vote(Xtr, ytr, Xte, k=3)
print(preds)  # [1, 1]
```

The first test point is closer to all three training points; the majority label among $\{0,1,1\}$ is $1$.  
The second test point is closest only to the third training point (label $1$), so its prediction is also $1$.

## What the gate checks

* **Correctness** – The predictions must match those of a reference implementation that uses the same deterministic tie‑breaking rule.  
  The grader computes the *argmax_agreement* between your output and the reference; it requires an exact agreement ($\text{score}=1.0$).  

No performance or style checks are applied, but the solution should be straightforward to understand.
