## Context

The **softmax** of a vector $x \in \mathbb{R}^n$ is

$$\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}.$$

Taking the log of both sides gives the **log-softmax**:

$$\operatorname{log\_softmax}(x_i) = x_i - \log\!\left(\sum_{j=1}^{n} e^{x_j}\right).$$

The inner sum $\operatorname{LSE}(x) = \log\!\left(\sum_{j} e^{x_j}\right)$ is
called the **LogSumExp**. A naïve implementation of LSE (just `math.log(sum(math.exp(v) for v in x))`)
suffers from overflow whenever any component of $x$ is large, because $e^{x_j}$
exceeds `float64` range for $x_j \gtrsim 710$.

The standard fix is the **stable LSE identity**:

$$\operatorname{LSE}(x) = m + \log\!\left(\sum_{j=1}^{n} e^{x_j - m}\right), \qquad m = \max_j\, x_j.$$

Subtracting the maximum before exponentiating guarantees every term $e^{x_j - m} \leq 1$,
so no overflow occurs. Combining the two identities:

$$\operatorname{log\_softmax}(x_i) = x_i - m - \log\!\left(\sum_{j} e^{x_j - m}\right).$$

## Task

Implement `log_softmax`:

```python
def log_softmax(x: list[float] | list[list[float]]) -> list[float] | list[list[float]]:
    ...
```

The input `x` is a list of arbitrary shape; compute log-softmax along the
**last axis** and return an array of the same shape with dtype `float64`.

You **must** use the stable $x - \operatorname{LSE}(x)$ identity. Do not compute
softmax first and then take its log — that approach overflows for large inputs and
will not pass the gate.

## Example

```python
x = [1.0, 2.0, 3.0]
result = log_softmax(x)
# result ≈ [-2.4076, -1.4076, -0.4076]
# sum of exp(result) ≈ 1.0
```

For large values the naïve approach breaks:

```python
x = [1000.0, 1001.0, 1002.0]
log_softmax(x)           # correct: [-2.4076, -1.4076, -0.4076]
[math.log(math.exp(v) / sum(math.exp(v) for v in x)) for v in x] # returns [-nan, -nan, -nan]
```

## What the gate checks

The gate computes a reference log-softmax using a separate stable LSE
implementation, then measures

$$\max_{i} \bigl|\,\text{your}[i] - \text{ref}[i]\,\bigr|.$$

The test suite includes vectors with extreme values ($\pm 10^3$), uniform
vectors, single-element vectors, and batched 2-D inputs. The absolute
error must be below $10^{-7}$ on every case.
