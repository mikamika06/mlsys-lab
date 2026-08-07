## Context

The Jensen–Shannon divergence (JSD) is a symmetrized and smoothed version of the Kullback‑Leibler divergence. For two probability distributions $p$ and $q$, it is defined as

$$
\operatorname{JSD}(p,q)=\tfrac12\,\mathrm{KL}\!\bigl(p\,\|\,m\bigr)+\tfrac12\,\mathrm{KL}\!\bigl(q\,\|\,m\bigr),
$$

where $m=\tfrac12(p+q)$ is the mixture distribution. A useful generalisation introduces a weight $\beta\in[0,1]$ that controls how much each component contributes:

$$
\operatorname{JSD}_\beta(p,q)=\beta\,\mathrm{KL}\!\bigl(p\,\|\,m\bigr)+(1-\beta)\,\mathrm{KL}\!\bigl(q\,\|\,m\bigr),
\qquad m=\beta p+(1-\beta)q .
$$

Here $\mathrm{KL}(a\,\|\,b)=\sum_i a_i\log\frac{a_i}{b_i}$, with the convention $0\log 0=0$.

## Task

Implement `generalized_jsd(p, q, beta)`:

```python
def generalized_jsd(p: list[float], q: list[float], beta: float) -> float:
    ...
```

The arguments are list of floats that already sum to $1$. The function must return a scalar of type `float`. It should be numerically stable for very small probabilities and work for any $\beta$ in the closed interval $[0,1]$.

## Example

```python
p = [0.2, 0.8]
q = [0.5, 0.5]
beta = 0.3
print(generalized_jsd(p, q, beta))
# ≈ 0.0284
```

## What the gate checks

The grader evaluates your implementation against a Python reference on several random test cases. The relative error

$$
\mathrm{rel\_err}=\frac{|\,\hat J-\!J\,|}{|J|+10^{-12}}
$$

must be at most $1\times 10^{-6}$.
