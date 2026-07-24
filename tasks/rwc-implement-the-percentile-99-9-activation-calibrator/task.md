## Context

Activation quantization needs a scale range that represents the typical magnitude of a tensor while ignoring rare extreme values. A percentile calibrator collects activation magnitudes and chooses an absolute maximum threshold from their distribution.

For collected activation values $X$, define the magnitude distribution

$$M = \{|x| : x \in X\}.$$

The percentile calibrator with percentile $p = 99.9$ selects

$$a_{\max} = \mathrm{percentile}_{99.9}(M).$$

Values larger than $a_{\max}$ are clipped before quantization. This reduces the effect of rare outliers while preserving most activation values.

## Task

Implement `percentile_amax(calibration_batches)`:

```python
def percentile_amax(calibration_batches):
    ...
```

`calibration_batches` is a list of NumPy arrays containing activation tensors collected during calibration. Flatten all batches, collect the absolute activation magnitudes, and return the 99.9th percentile as a Python `float`.

The implementation should use NumPy operations. The returned value must match NumPy's percentile calculation using the combined $|X|$ distribution.

## Example

```python
import numpy as np

batches = [
    np.array([-2.0, 1.0, 4.0]),
    np.array([0.5, -10.0])
]

amax = percentile_amax(batches)
# equivalent to:
# np.percentile(np.abs(np.concatenate(batches)), 99.9)
```

## What the gate checks

The gate builds calibration datasets and computes the oracle value with NumPy:

$$a_{\max}^{\mathrm{oracle}} =
\mathrm{percentile}_{99.9}(|X|).$$

The returned `amax` is checked with relative error

$$\mathrm{rel\_err} =
\frac{|a_{\max} - a_{\max}^{\mathrm{oracle}}|}
{|a_{\max}^{\mathrm{oracle}}| + 10^{-12}}.$$

The gate also applies the returned threshold to an int8-style clipping simulation and checks that the resulting quantization error matches the oracle threshold's error. Implementations that use max, mean, or a different percentile will fail.
