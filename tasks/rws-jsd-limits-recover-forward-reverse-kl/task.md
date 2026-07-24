## Context

The Jensen-Shannon divergence is a smoothed comparison between two probability
distributions. For a skew parameter $\beta \in (0,1)$, define the mixture

$$
M_\beta = \beta q + (1-\beta)p .
$$

The normalized skew Jensen-Shannon divergence is

$$
\operatorname{JSD}_\beta(p,q)
=
\frac{
\beta \operatorname{KL}(q \Vert M_\beta)
+
(1-\beta)\operatorname{KL}(p \Vert M_\beta)
}{
\beta(1-\beta)
}.
$$

The normalization makes the endpoint behavior recover KL divergences. As
$\beta \rightarrow 1$, the mixture approaches $q$ and the expression approaches
the forward KL divergence from $p$ to $q$:

$$
\lim_{\beta \rightarrow 1} \operatorname{JSD}_\beta(p,q)
=
\operatorname{KL}(p\Vert q).
$$

As $\beta \rightarrow 0$, it approaches the reverse KL divergence:

$$
\lim_{\beta \rightarrow 0} \operatorname{JSD}_\beta(p,q)
=
\operatorname{KL}(q\Vert p).
$$

For two discrete distributions with positive probabilities,

$$
\operatorname{KL}(a\Vert b)
=
\sum_i a_i \log\frac{a_i}{b_i}.
$$

## Task

Implement `jsd_beta(p, q, beta)`:

```python
def jsd_beta(p: np.ndarray, q: np.ndarray, beta: float) -> float:
    ...
```

The function receives two one-dimensional NumPy arrays containing probability
distributions and a scalar $\beta$ strictly between $0$ and $1$. Return the
normalized skew Jensen-Shannon divergence as a Python float.

Assume `p` and `q` have the same length, contain positive values, and each sum
to $1$. Use NumPy operations for the calculation.

## Example

```python
import numpy as np

p = np.array([0.8, 0.2])
q = np.array([0.5, 0.5])

value = jsd_beta(p, q, 0.999999)
# The result is close to KL(p || q)
```

## What the gate checks

The gate computes a NumPy reference implementation of the normalized
Jensen-Shannon formula. It tests distributions with beta values close to both
endpoints and compares the learner result against the oracle using relative
error.

The metric `rel_err` is the mean relative error over all tested values. The
gate requires

$$
\mathrm{rel\_err} \le 10^{-4}.
$$
