## Context

TensorRT INT8 entropy calibration chooses a clipping threshold by comparing the
original activation distribution with a lower-resolution quantized
approximation. The calibration histogram represents activation magnitudes
$|X|$.

The histogram contains $2048$ bins. For a candidate threshold bin $T$, all bins
above $T$ are clipped into the last retained bin. The retained histogram is then
requantized into $128$ bins and expanded back to the retained resolution.

The quality of a threshold is measured using KL divergence:

$$
D_{KL}(P \Vert Q) = \sum_i P_i \log\frac{P_i}{Q_i}.
$$

The selected threshold is the candidate minimizing this divergence.

## Task

Implement:

```python
def entropy_calibration_threshold(hist: np.ndarray) -> tuple[int, np.ndarray]:
    ...
```

The input `hist` is a one-dimensional NumPy array with exactly $2048$ histogram
counts.

For every candidate threshold $T$ from $128$ through $2047$:

1. Clip the histogram by adding all bins after $T$ into bin $T$.
2. Treat the clipped histogram as the reference distribution $P$.
3. Divide the clipped histogram into $128$ equal quantization regions.
4. Replace every region with its average count to form the expanded quantized
   distribution $Q$.
5. Compute the KL divergence between $P$ and $Q$.

Return:

- the threshold index with the smallest KL divergence;
- a `float64` NumPy array of length $1920` containing the KL divergence values
  in candidate order.

## Example

```python
import numpy as np

hist = np.zeros(2048, dtype=np.float64)
hist[:8] = [10, 20, 40, 80, 60, 30, 15, 5]

threshold, curve = entropy_calibration_threshold(hist)

# curve[0] corresponds to threshold bin 128
# curve[-1] corresponds to threshold bin 2047
```

## What the gate checks

The gate builds an independent NumPy oracle implementing the clipping,
requantization, expansion, and KL divergence calculation.

The returned threshold must match the oracle's argmin. The KL curve is compared
using:

$$
\mathrm{rel\_err} =
\frac{\lVert c_{\mathrm{student}}-c_{\mathrm{oracle}}\rVert_2}
{\lVert c_{\mathrm{oracle}}\rVert_2+10^{-12}}.
$$

The curve must satisfy $\mathrm{rel\_err}\leq 10^{-9}$.
