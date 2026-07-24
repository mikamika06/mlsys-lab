## Context
Softmax is a widely used function that maps a vector of logits into a probability distribution. The behavior of softmax can be controlled by a hyperparameter called the temperature ($T$). 
The temperature-scaled softmax is defined as:
$$ P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} $$

As $T \to \infty$, the distribution becomes uniform.
As $T \to 0$, the distribution approaches the argmax (a one-hot vector).

However, directly computing the exponential of large numbers can lead to overflow. When scaling by a very small temperature $T$, the logits $z_i / T$ can become extremely large. To ensure numerical stability, it is standard practice to subtract the maximum value before exponentiation.

## Task
Implement a function `compute_softmax(logits, temperatures)` that computes the temperature-scaled softmax for a 1D vector of `logits` across multiple `temperatures`.

The function must return a list of lists containing the probabilities. The outer list corresponds to each temperature in the `temperatures` array, and the inner list corresponds to the resulting probabilities for the `logits`.

**Important**: Your implementation must be numerically stable and avoid overflow even for very small values of $T$ (e.g., $T=10^{-6}$).

## Example
```python
logits = [1.0, 2.0, 3.0]
temperatures = [1.0, 0.5]
compute_softmax(logits, temperatures)
# [
#   [0.09003057, 0.24472847, 0.66524096],
#   [0.01714783, 0.12672474, 0.85612744]
# ]
```

## What the gate checks
- `mean_kl`: Mean Kullback-Leibler (KL) divergence between your computed probabilities and the reference probabilities across all temperatures. It should be $\le 10^{-9}$. We test with temperatures as low as $10^{-6}$.
