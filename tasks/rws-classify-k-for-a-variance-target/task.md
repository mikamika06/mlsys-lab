## Context

In many dimensionality‑reduction or feature‑selection pipelines one needs to decide how many principal components (or eigenvectors) are required to explain a given fraction of the total variance.  
Let $\lambda_1,\dots,\lambda_n$ be the eigenvalues of a covariance matrix sorted in non‑increasing order, and let $T\in(0,1]$ be a target proportion of explained variance.  The cumulative variance ratio after selecting the first $k$ components is

$$
R_k = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{n}\lambda_j}.
$$

The task is to find the smallest integer $k$ such that $R_k \ge T$.  This $k$ tells us how many components are needed to reach or exceed the target variance.

## Task

Implement the function

```python
def classify_k_for_variance_target(eigenvalues: list[float], target: float) -> int:
    ...
```

* `eigenvalues` is a list of floats of shape `(n,)` containing eigenvalues sorted in non‑increasing order.
* `target` is a float in the interval $(0,1]$ representing the desired cumulative variance proportion.
* The function must return an integer $k$ with $1 \le k \le n$ that satisfies
  $$\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{n}\lambda_j} \ge target$$
  and is minimal among all such integers.

The implementation should be straightforward, using only Python operations; no explicit Python loops are required but they are allowed if the solution still passes the gate.

## Example

```python
eigs = [4.0, 2.0, 1.0, 0.5]
target = 0.8
k = classify_k_for_variance_target(eigs, target)
print(k)          # → 3
```

Explanation:  
Total variance $=7.5$.  
Cumulative sums: $[4.0, 6.0, 7.0, 7.5]$ giving ratios $[0.533, 0.8, 0.933, 1.0]$.  
The smallest $k$ with ratio $\ge 0.8$ is $3$.

## What the gate checks

A single metric `argmin_index` compares the returned index to a Python‑based oracle that computes the minimal $k$ exactly as described above.  The solution passes if and only if the two indices match for all test cases.  No other performance or style constraints are enforced.
