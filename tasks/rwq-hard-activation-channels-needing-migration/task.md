## Context

When quantizing neural-network activations to `int8`, each channel $j$ (i.e.\ each column of the activation matrix $X \in \mathbb{R}^{n \times d}$) is scaled to fit the $[-127, 127]$ range. If a channel contains a few very large values — so-called *outlier channels* — the per-tensor quantization scale is dominated by that channel, and every other channel loses precision.

A common production strategy (used in e.g.\ LLM.int8 and bitsandbytes mixed-precision kernels) is to **detect** these hard channels and **migrate** them to a higher-precision format (fp16) while keeping the remaining channels in int8.

Detection works as follows. For each channel $j$, compute the absolute maximum activation:

$$\operatorname{absmax}_j = \max_{i} |X_{ij}|.$$

Then compute a robust reference scale across all channels — the **median** of the per-channel absmax values:

$$\rho = \operatorname{median}(\operatorname{absmax}_1, \ldots, \operatorname{absmax}_d).$$

A channel $j$ is flagged for migration when

$$\operatorname{absmax}_j > \tau \cdot \rho,$$

where $\tau$ is a caller-specified threshold (typically in $[2, 5]$).

Using the median rather than the mean makes the reference robust: the very outliers we are trying to detect do not pull $\rho$ upward.

## Task

Implement `migration_channels`:

```python
import numpy as np

def migration_channels(X: np.ndarray, threshold: float) -> list[int]:
    ...
```

**Parameters:**

- `X` — a 2-D NumPy array of shape $(n, d)$ containing float64 activations.
- `threshold` — a positive float $\tau$.

**Returns:**

A sorted list of integer channel indices (zero-based) whose per-channel `absmax` exceeds $\tau \cdot \rho$, where $\rho$ is the median of all per-channel absmax values.

Use NumPy vectorised operations. Do **not** write Python `for` loops over rows.

## Example

```python
import numpy as np

X = np.array([
    [ 0.5,  1.0,  0.1],
    [-0.3,  2.0, -0.2],
    [ 0.8, 50.0,  0.4],
    [-0.1, -0.5, -0.3],
])
# per-channel absmax: [0.8, 50.0, 0.4]
# median rho = 0.8
# with threshold 3.0: cutoff = 2.4
# only channel 1 (50.0 > 2.4) is flagged
migration_channels(X, 3.0)  # -> [1]
```

## What the gate checks

The grader generates several test activation matrices of varying shapes and outlier patterns, calls your function on each, and compares the returned sorted index list against a NumPy oracle that independently computes the per-channel `absmax`, the median, and the threshold comparison. Any difference — wrong axis, wrong aggregation, wrong comparison operator, off-by-one — results in `exact_match = 0.0`.
