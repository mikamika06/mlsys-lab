## Context

In activation‑aware weight quantization (AWQ) the per‑channel scale is used to normalise activations before they are multiplied by quantised weights.  
For a calibration tensor $X \in \mathbb{R}^{N\times C}$, where $N$ is the number of samples and $C$ the number of channels, a common choice for the scale vector $s\in\mathbb{R}^C$ is

$$
s_c = \frac{\lVert X_{\cdot,c}\rVert_2}{\sqrt{N}}
= \frac{\sqrt{\sum_{n=1}^{N} X_{nc}^2}}{\sqrt{N}}\;,
$$

which represents the root‑mean‑square (RMS) magnitude of each channel.

## Task

Implement `per_channel_scales(X)`:

```python
def per_channel_scales(X: np.ndarray) -> np.ndarray:
    ...
```

The function receives a 2‑D NumPy array $X$ of shape $(N, C)$ and must return a 1‑D float64 array of length $C$ containing the RMS scale for each channel.  
Use only vectorised NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
X = np.array([[0, 2], [3, 4]])
# Channel 0: sqrt(0^2 + 3^2) / sqrt(2) = 3/√2 ≈ 2.12132
# Channel 1: sqrt(2^2 + 4^2) / sqrt(2) = √20 / √2 = √10 ≈ 3.16228
s = per_channel_scales(X)
print(s)   # [2.12132034 3.16227766]
```

## What the gate checks

The grader computes a reference scale vector using NumPy and compares it to your output with the `channel_rel_err` scorer from `arena.scorers`.  
Your implementation must achieve a relative error $\le 10^{-9}$ on all test cases.
