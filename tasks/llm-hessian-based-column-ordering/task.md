## Context

In post‑training quantization (GPTQ) the order in which weight columns are updated influences the final error. A common heuristic is to rank columns by their *diagonal Hessian* with respect to a loss function; columns that have larger second‑order sensitivity are treated first. For a linear layer \(y = Wx\) and a squared‑error loss, the diagonal of the Hessian with respect to each weight element can be shown to be proportional to \(\sum_{s} x_s^2\). Aggregating over all output units gives a salience score for each input dimension

$$
h_j = \Bigl(\sum_{i=1}^{n_{\text{batch}}} x_{ij}^2\Bigr)\,
      \Bigl(\sum_{k=1}^{n_{\text{out}}} w_{kj}^2\Bigr),
$$

where \(x_{ij}\) is the activation of input dimension \(j\) in sample \(i\), and \(w_{kj}\) is the weight connecting that input to output unit \(k\).

## Task

Implement `hessian_saliency(W, A)`:

```python
def hessian_saliency(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    ...
```

`W` is a 2‑D NumPy array of shape \((n_{\text{out}}, n_{\text{in}})\) containing the weight matrix of a linear layer.  
`A` is a 2‑D NumPy array of shape \((n_{\text{batch}}, n_{\text{in}})\) containing a batch of activations that were fed to the layer during inference.

The function must return a 1‑D float64 array `h` of length \(n_{\text{in}}\) where each entry is the diagonal Hessian salience for the corresponding input dimension, computed as

$$
h_j = \Bigl(\sum_{i} A_{ij}^{2}\Bigr)\,
      \Bigl(\sum_{k} W_{kj}^{2}\Bigr).
$$

The implementation must use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
W = np.array([[1, 2], [3, 4]], dtype=np.float64)
A = np.array([[5, 6], [7, 8]], dtype=np.float64)

h = hessian_saliency(W, A)
# h ≈ [ (5^2+7^2)*(1^2+3^2) , (6^2+8^2)*(2^2+4^2) ]
#   = [ (25+49)*(1+9) , (36+64)*(4+16) ]
#   = [ 74*10 , 100*20 ] = [740, 2000]
```

## What the gate checks

The grader computes a reference salience vector using NumPy and compares it to your output with the global relative L2 error

$$
\mathrm{rel\_err} = \frac{\lVert h_{\text{cand}} - h_{\text{ref}}\rVert}{\lVert h_{\text{ref}}\rVert}.
$$

Your solution must achieve $\mathrm{rel\_err}\le 10^{-9}$ on a set of random test cases. A fully vectorised implementation that follows the formula above will pass; any loop‑based or incorrect computation will fail.
