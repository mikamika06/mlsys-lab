## Context

In many neural‑network models the magnitude of a channel’s activations can indicate its importance for the forward pass. A simple saliency score for channel $c$ is the mean absolute activation over all samples and spatial locations,

$$s_c = \frac{1}{N}\sum_{n=1}^{N} \lvert X_n^{(c)}\rvert,$$

where $X_n^{(c)}$ denotes the tensor of activations for channel $c$ in sample $n$. Channels with a large $s_c$ are considered *salient*.

A common heuristic is to keep only the top fraction of channels. In this task we keep the **top 1 %** of channels by saliency.

## Task

Implement `identify_salient_channels(X)`:

```python
def identify_salient_channels(X: np.ndarray) -> list[int]:
    ...
```

`X` is a NumPy array whose second dimension corresponds to the channel axis (e.g. shape `(batch, channels, height, width)`). The function must return a **sorted** list of integer indices of the channels that belong to the top 1 % by saliency. If the number of channels `C` does not divide evenly into 100, round up: `k = ceil(C × 0.01)`.

The implementation must use only NumPy operations; no explicit Python loops over channels are allowed.

## Example

```python
import numpy as np
X = np.array([
    [[1, -2], [3, 4]],
    [[-1, 5], [6, -7]]
])  # shape (2, 2, 2)

# saliency per channel:
# channel 0: mean(|[1,-2]| + |-1,5|) = ...
# channel 1: ...

indices = identify_salient_channels(X)
print(indices)   # e.g. [1]
```

## What the gate checks

The grader computes a reference set of top‑1 % channels from a deterministic random tensor and compares it to the student’s output using an **exact match** metric. The returned list must contain exactly the same indices, sorted in ascending order.

No other metrics are checked; however, the implementation should be fully vectorized for clarity.
