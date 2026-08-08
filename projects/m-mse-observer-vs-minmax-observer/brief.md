Building integer quantization pipelines requires choosing how we map floating-point values into limited integer ranges. The mapping is governed by two parameters: a `scale` and a `zero_point`.

The simplest way to find these is the **MinMax Observer**: it takes the strict minimum and maximum of the tensor and stretches the integer range to fit them. This is fast, but a single outlier ruins the precision for every other value in the tensor. The alternative is the **MSE Observer**, which sweeps a range of candidate scales and chooses the one that minimizes the mean squared error (MSE) between the original and dequantized tensors.

A common optimization when performing quantized matmuls or sums is to ignore the zero-point entirely during the inner loop to save cycles, attempting to correct for it later.

In this unit, you will build a quantization observer toolkit:
1. **Parse schemas:** Convert string scheme names like `"int8-sym"` into quantization arguments.
2. **MinMax vs MSE:** Implement both observers. For the MSE observer, sweep 100 candidate scales uniformly spaced by applying a multiplier `alpha` from `0.1` to `1.0` (inclusive) to the MinMax scale. Pick the candidate that minimizes MSE, retaining the MinMax zero-point.
3. **Bias Quantification:** Implement a function that calculates the exact numerical bias introduced by dequantizing a tensor while ignoring its zero-point, demonstrating why zero-point compensation is mandatory for asymmetric schemes.

Finally, write regression tests verifying key mathematical invariants: the MSE observer must never yield a worse error than MinMax, and the ignored zero-point bias must grow linearly with the tensor size.
