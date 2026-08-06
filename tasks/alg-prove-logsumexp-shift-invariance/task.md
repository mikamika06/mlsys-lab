## Context

The **log‑sum‑exp** (LSE) function is a numerically stable way to compute the logarithm of a sum of exponentials:

$$\operatorname{lse}(x)=\log\!\Bigl(\sum_{i} e^{x_i}\Bigr).$$

A key property of LSE is *shift invariance*:

$$\operatorname{lse}(x+c)-c=\operatorname{lse}(x),$$

for any scalar shift $c$.  This follows directly from the definition because adding a constant to every component simply factors out of the exponential sum.

In practice, computing $\operatorname{lse}$ with naive `math.log(sum(math.exp(v) for v in x))` can overflow or underflow for large magnitude inputs. A stable implementation first subtracts the maximum element:

$$\operatorname{lse}(x)=m+\log\!\Bigl(\sum_{i} e^{\,x_i-m}\Bigr), \qquad m=\max_i x_i.$$

## Task

Implement a function `logsumexp(x: list[float]) -> float` that returns the log‑sum‑exp of a 1‑D list.  
The implementation must be fully vectorised, use only Python, and avoid overflow/underflow for inputs with large magnitude.

```python
def logsumexp(x: list[float]) -> float:
    ...
```

## Example

```python
x = [0.0, 1.0, 2.0]
print(logsumexp(x))
# 2.4076059644443806
```

The result matches the exact value $\log( e^0 + e^1 + e^2 )$.

## What the gate checks

Two random test cases are generated:

* A moderate‑size array with values in $[-10,\,10]$.
* An array containing very large positive and negative numbers (e.g. scaled by 1000).

For each case a random shift $c$ is chosen.  
The grader computes the reference value using Python’s `math.log` and `sum`.
It then calls your implementation twice:

1. $\hat y_1 = \texttt{logsumexp}(x)$
2. $\hat y_2 = \texttt{logsumexp}(x+c)-c$

The gate measures the maximum absolute error

$$\max_{\ell=1,2}\bigl|\hat y_\ell-\operatorname{lse}_\ell\bigr|,$$

and requires it to be at most $10^{-10}$.

A correct implementation will pass both cases; a naive overflow‑prone version will fail the shift‑invariance test.
