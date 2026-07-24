## Context

In many neural‑network libraries the scale used to normalise activations is updated during a calibration phase.  
A common strategy is **moving absolute maximum** (absmax): for each tensor we keep a running estimate of the largest absolute value that has appeared so far, and update it with momentum $m \in [0,1]$:

$$
s_{t+1} = m\, s_t + (1-m)\,\max_{\mathbf{x}\in B_t}\lVert\mathbf{x}\rVert_\infty ,
$$

where $B_t$ is the batch at step $t$.  
The momentum term smooths the estimate across batches, preventing large spikes from dominating the scale.

## Task

Implement a function that performs this moving‑absmax calibration over a sequence of activation batches.

```python
import numpy as np
from typing import List

def moving_absmax(batches: List[np.ndarray], momentum: float) -> np.ndarray:
    """
    Compute a per‑tensor running absolute maximum with momentum.

    Parameters
    ----------
    batches : list of 2‑D arrays, shape (batch_size, num_tensors)
        Activation values for each tensor in successive calibration batches.
    momentum : float
        Momentum coefficient $m$ used to update the running scale.

    Returns
    -------
    scales : np.ndarray, dtype=float64, shape (num_tensors,)
        Final per‑tensor absolute maximum after processing all batches.
    """
    ...
```

The function must:

1. Work for any number of tensors (`num_tensors` is the second dimension of each batch).
2. Use only NumPy operations; no Python loops over elements or tensors.
3. Return a `float64` array.

## Example

```python
import numpy as np

batches = [
    np.array([[ 1, -2], [ 3,  4]]),   # batch 0
    np.array([[-5,  6], [ 7, -8]])    # batch 1
]
momentum = 0.9

scales = moving_absmax(batches, momentum)
print(scales)          # array([ 5.,  8.])
```

The first tensor’s running absmax is  
$0 \xrightarrow{m=0.9} 0.1\times\max(|1|,|3|)=0.3$  
then updated with batch 1: $0.9\times0.3+0.1\times7=0.87$, rounded to $5.$ in this toy example.

## What the gate checks

The grader generates random batches and a momentum value, computes a reference implementation using NumPy, and evaluates the relative L2 error:

$$
\mathrm{rel\_err} = \frac{\lVert \hat{s}-s\rVert_2}{\lVert s\rVert_2 + 10^{-12}} .
$$

The candidate must achieve $\mathrm{rel\_err}\le 1\times10^{-6}$ and return a `float64` array.  
Any deviation (wrong update rule, wrong dtype, or runtime error) will cause the gate to fail.
