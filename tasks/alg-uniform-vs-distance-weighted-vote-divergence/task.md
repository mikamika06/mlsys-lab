## Context

k‑Nearest Neighbours (kNN) is a non‑parametric classification method that assigns to each query point the most common label among its $k$ nearest training points.  
Two popular voting schemes are:

* **Uniform vote** – every neighbour contributes one vote to its class.
* **Distance‑weighted vote** – neighbours contribute a weight inversely proportional to their distance:
  $$w_i = \frac{1}{d_i + \varepsilon},$$
  where $d_i$ is the Euclidean distance from the query point to neighbour $i$ and $\varepsilon>0$ prevents division by zero.

When several classes receive an equal number of votes (or equal weighted sum), the class with the smallest integer label wins. This deterministic tie‑breaking rule guarantees reproducible results across platforms.

## Task

Implement a function that, given training data $(X_{\text{train}}, y_{\text{train}})$, a set of query points $X_{\text{test}}$, and an integer $k$, returns two arrays:

```python
def knn_vote_divergence(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test:  np.ndarray,
                        k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `X_train` – shape $(n_{\text{train}}, d)$, float64.
* `y_train` – shape $(n_{\text{train}},)$, integer class labels (non‑negative).
* `X_test`  – shape $(n_{\text{test}}, d)$, float64.
* `k`       – number of neighbours to consider ($1 \le k \le n_{\text{train}}$).

The function must return a tuple `(uniform_labels, weighted_labels)` where each element is an integer array of length $n_{\text{test}}$.  
Use only NumPy operations; avoid explicit Python loops over the test set. The implementation should be deterministic and work on any platform.

## Example

```python
import numpy as np

X_train = np.array([[0., 0.],
                    [1., 0.],
                    [0., 1.],
                    [1., 1.]])
y_train = np.array([0, 1, 1, 0])

X_test = np.array([[0.2, 0.2],
                   [0.8, 0.8],
                   [0.5, 0.5]])

uniform, weighted = knn_vote_divergence(X_train, y_train, X_test, k=3)

print(uniform)   # [0 1 1]
print(weighted)  # [0 1 1]  (in this toy example both schemes agree)
```

## What the gate checks

The grader computes reference predictions for both voting schemes using a trusted NumPy‑based implementation.  
Two metrics are evaluated:

* **uniform_agreement** – `argmax_agreement` between the user’s uniform labels and the reference uniform labels.
* **weighted_agreement** – `argmax_agreement` between the user’s weighted labels and the reference weighted labels.

Both metrics must equal `1.0`.  The test data contains points where the two schemes produce different predictions, so a correct implementation will return distinct arrays for the two voting methods.  
The grader also verifies that the returned arrays have the expected shape and dtype (`int64`).
