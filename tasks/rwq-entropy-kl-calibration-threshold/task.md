## Context

Post-training quantization calibrators often select an activation clipping threshold by minimizing the divergence between the original activation distribution and a quantized approximation.

Given activation samples $x$, a histogram with $B$ bins approximates the original probability distribution:

$$
p_i = \frac{h_i}{\sum_j h_j},
$$

where $h_i$ is the count in histogram bin $i$.

For a candidate threshold, values above the threshold are clipped and the remaining range is represented using fewer quantization levels. The reconstructed histogram defines a distribution $q$. The calibration objective is the KL divergence:

$$
D_{KL}(p \parallel q) = \sum_i p_i \log \frac{p_i}{q_i}.
$$

The entropy calibration method evaluates several candidate thresholds and chooses the one with the smallest KL divergence.

## Task

Implement `calibrate_threshold_index(activations, num_bins, candidate_indices)`.

The function receives:

- `activations`: a 1-D NumPy array of non-negative floating point activation values.
- `num_bins`: the fixed number of histogram bins.
- `candidate_indices`: a 1-D sequence of histogram bin indices to evaluate.

Build a histogram over the range from $0$ to the maximum activation value. For every candidate index $k$:

1. Use the histogram edge at index $k$ as the clipping threshold.
2. Clip activation values above that threshold.
3. Uniformly quantize the clipped values into $k$ levels.
4. Reconstruct the quantized values using the center of each quantization level.
5. Compute the KL divergence between the original histogram distribution and the reconstructed histogram distribution.

Return the candidate index with the smallest KL divergence as a Python integer.

Use deterministic NumPy operations only.

The required function signature is:

```python
def calibrate_threshold_index(
    activations: np.ndarray,
    num_bins: int,
    candidate_indices: np.ndarray
) -> int:
    ...
```

## Example

```python
import numpy as np

x = np.array([0.1, 0.4, 0.8, 1.2, 2.0, 2.1])
idx = calibrate_threshold_index(
    x,
    num_bins=16,
    candidate_indices=np.array([4, 8, 12])
)
# idx is the candidate histogram index with the lowest KL divergence
```

## What the gate checks

The gate builds its own NumPy oracle implementing the histogram, threshold sweep, quantization, and KL computation.

The returned `argmin_index` must exactly match the oracle-selected candidate. The reported `kl_error` is the absolute difference between the candidate KL value and the oracle KL value for the selected threshold and must satisfy $kl\_error \le 10^{-6}$.
