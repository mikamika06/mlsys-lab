## Context

Training systems often change how examples are sampled. A uniform sampler gives every
domain the same probability, while a reweighted sampler can emphasize domains with
higher expected loss reduction.

Assume each domain $i$ has a loss model parameterized by a weight variable $x$:

$$
L_i(x) = a_i + b_i x + c_i x^2 .
$$

For this task, the current loss estimate is evaluated at $x=1$:

$$
\ell_i = L_i(1) = a_i + b_i + c_i .
$$

The uniform expected loss over $n$ domains is

$$
E_{\mathrm{uniform}} = \frac{1}{n}\sum_{i=1}^{n}\ell_i .
$$

The reweighted sampler uses a simplex of probabilities generated from a loss-aware
softmax:

$$
p_i = \frac{e^{-\ell_i}}{\sum_j e^{-\ell_j}} .
$$

The expected loss under the reweighted distribution is

$$
E_{\mathrm{reweighted}} = \sum_{i=1}^{n} p_i \ell_i .
$$

The reduction is the difference between the two expectations:

$$
R = E_{\mathrm{uniform}} - E_{\mathrm{reweighted}} .
$$

## Task

Implement `compare_sampling(coeffs)`.

The input is a list of shape $(n, 3)`. Each row contains the coefficients
$[a_i, b_i, c_i]$ for one domain's quadratic loss model.

Return a tuple:

```python
(
    uniform_loss,
    reweighted_loss,
    reduction
)
```

where all three values are Python floats. Use Python operations for the computation.
The softmax calculation must be numerically stable by shifting values before
exponentiation.

## Example

```python

coeffs = [
    [1.0, 0.5, 0.0],
    [3.0, 0.0, 0.0],
]

uniform_loss, reweighted_loss, reduction = compare_sampling(coeffs)

# The two domain losses are 1.5 and 3.0.
# uniform_loss = 2.25
# reweighted_loss is closer to 1.5 because the lower-loss domain gets more weight.
```

## What the gate checks

The gate builds several coefficient matrices and computes the oracle values using
the quadratic loss model and a Python softmax implementation. The returned uniform
loss, reweighted loss, and reduction must each have relative error at most
$10^{-12}$ compared with the oracle.
