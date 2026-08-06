## Context

ALiBi (Attention with Linear Biases) adds a head-specific linear bias to attention
scores. Each attention head $h$ uses a slope $m_h$ that is multiplied by the
relative position distance.

For $n$ attention heads, the slopes should follow a geometric schedule. The base
ratio is determined by the number of heads:

$$
m_h = 2^{-8h/n}, \quad h = 0,1,\dots,n-1 .
$$

A common implementation bug is confusing the exponent with the base. Computing
$2^{-8}/n$ and then raising values from that ratio produces a very different
sequence than computing the exponent $-8h/n$ directly.

The correct sequence should start at $1$ and decrease geometrically toward
$2^{-8}$ across the heads.

## Task

Implement `alibi_slopes(n)`:

```python
def alibi_slopes(n: int) -> list[float]:
    ...
```

Return a list of shape $(n,)$ with dtype `float64` containing the ALiBi
slopes for $n$ attention heads.

The returned values must satisfy

$$
\mathrm{slope}[h] = 2^{-8h/n}.
$$

Assume $n$ is a positive integer.

## Example

```python

slopes = alibi_slopes(4)

# array([1.        , 0.25      , 0.0625    , 0.015625])
```

## What the gate checks

The gate computes an independent Python reference implementation of the geometric
formula and compares the submitted result using maximum absolute error:

$$
\max_h |m_h^{candidate} - m_h^{reference}|.
$$

The error must be below $10^{-6}$. Implementations using the incorrect ratio
$2^{-8}/n$ instead of the exponent $-8h/n$ fail this check.
