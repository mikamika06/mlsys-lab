## Context

The soft‑max function maps a vector of logits $z \in \mathbb{R}^k$ to a probability distribution

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{k}\exp(z_j)}.$$

Its logarithm, the *log‑softmax*, is often used in cross‑entropy loss because it avoids an intermediate division:

$$\operatorname{log\_softmax}(z)_i
   = z_i - \log\!\Bigl(\sum_{j=1}^{k}\exp(z_j)\Bigr).$$

When the logits contain large positive or negative values, computing $\exp(z)$ directly can overflow or underflow. A standard trick is to subtract the maximum logit before exponentiation:

$$
\operatorname{log\_softmax}(z)_i
= z_i - \bigl(\max_j z_j + \log\!\sum_{j}\exp(z_j-\max_j z_j)\bigr).
$$

This form is numerically stable for any real input.

## Task

Implement the function `log_softmax` that takes a 2‑D list of shape $(n, k)$ and returns an array of the same shape containing the log‑softmax values computed in a numerically stable way. The result must be of type `float64`.

```python
def log_softmax(x: list[list[float]]) -> list[list[float]]:
    ...
```

## Example

```python
x = [[0, 1, 2],
              [1000, 1000, 1000],
              [-1000, -999, -998]]
y = log_softmax(x)
print(y)  # [[-2.4076059644443806, -1.4076059644443806, -0.4076059644443806], [-1.0986122886681642, -1.0986122886681642, -1.0986122886681642], [-2.4076059644444285, -1.4076059644444285, -0.40760596444442854]]
```

## What the gate checks

The grader computes a reference log‑softmax using Python’s stable formulation and compares it to your output with the metric `max_abs_err`. The candidate passes only if

$$\mathrm{max\_abs\_err} \le 10^{-9}.$$

Additionally, the function must return a `float64` array of the same shape as the input. A naive implementation that first computes soft‑max and then takes the logarithm will overflow for large logits and fail this gate.
