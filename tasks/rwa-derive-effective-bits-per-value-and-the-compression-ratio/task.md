## Context

Weight quantization reduces storage by representing each value with fewer bits than a baseline format such as FP16. However, quantized storage also needs metadata such as per-group scales and zero points.

For a group of $g$ values, suppose each value uses $n_\mathrm{bits}$ bits. If each group stores a scale using $s_\mathrm{bits}$ bits and a zero point using $z_\mathrm{bits}$ bits, the amortized storage cost per value is

$$
\mathrm{effective\_bpv}
=
n_\mathrm{bits}
+
\frac{s_\mathrm{bits}+z_\mathrm{bits}}{g}.
$$

The compression ratio compared with FP16 is the quantized bits per value divided by the FP16 cost of $16$ bits:

$$
\mathrm{compression\_ratio}
=
\frac{\mathrm{effective\_bpv}}{16}.
$$

A lower ratio means the quantized representation uses fewer bits relative to FP16.

## Task

Implement `effective_bits_per_value(nbits, group_size, scale_bits=16, zero_bits=0)`:

```python
def effective_bits_per_value(nbits, group_size, scale_bits=16, zero_bits=0):
    ...
```

The function receives integer bit widths and a group size. It must return a tuple:

```python
(effective_bpv, compression_ratio)
```

where both values are Python floats.

Compute the amortized bits per value using the metadata overhead formula above, then compute the ratio against FP16 storage.

## Example

```python
effective_bits_per_value(4, 128, 16, 0)
# (4.125, 0.2578125)
```

## What the gate checks

The gate uses a NumPy oracle to compute the expected effective bits per value and compression ratio from several combinations of value bit widths, group sizes, and metadata sizes.

Both returned values must have relative error $\le 10^{-9}$ compared with the oracle result. A solution that ignores metadata overhead or uses the wrong FP16 reference size will fail.
