## Context

Training memory is not only the size of the model weights. A simple mixed-precision
training estimate can include parameters, gradients, and optimizer state.

For a model with $N$ parameters stored as `float32`, the parameter storage is

$$M_{\mathrm{param}} = N \cdot \mathrm{sizeof}(\mathrm{float32}).$$

If gradients use the same precision and an optimizer stores two additional
`float32` tensors per parameter, the total byte estimate is

$$M_{\mathrm{total}} =
N \cdot
\left(
\mathrm{sizeof}(\mathrm{float32})
+
\mathrm{sizeof}(\mathrm{float32})
+
2 \cdot \mathrm{sizeof}(\mathrm{float32})
\right)
+
M_{\mathrm{activation}} .
$$

The activation term for a batch can be estimated from the number of stored
activation elements:

$$M_{\mathrm{activation}} =
B \cdot S \cdot H \cdot L \cdot \mathrm{sizeof}(\mathrm{float32}),$$

where $B$ is batch size, $S$ is sequence length, $H$ is hidden size, and $L$ is
the number of layers.

A device with capacity $G$ gigabytes can run the configuration when

$$M_{\mathrm{total}} \leq G \cdot 10^9.$$

## Task

Implement `classify_feasibility(config, device_gb)`:

```python
def classify_feasibility(config: dict, device_gb: float) -> bool:
    ...
```

The `config` dictionary contains integer fields:

- `params`: number of model parameters
- `batch`: batch size
- `seq`: sequence length
- `hidden`: hidden size
- `layers`: number of layers

Return `True` when the estimated training memory fits on the device and
`False` when it would cause an out-of-memory condition.

Use NumPy dtypes to obtain storage sizes rather than assuming a byte width.

## Example

```python
config = {
    "params": 1000000,
    "batch": 2,
    "seq": 16,
    "hidden": 32,
    "layers": 4,
}

classify_feasibility(config, 1.0)
# True
```

## What the gate checks

The gate evaluates several model configurations and compares the returned boolean
with an independent reference calculation. The reference derives tensor storage
sizes from NumPy dtype information and applies the memory accounting model.

The `exact_match` score must be $1.0$.
