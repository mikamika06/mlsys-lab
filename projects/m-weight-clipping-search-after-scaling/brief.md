The final step in Activation-aware Weight Quantization (AWQ) is actually quantizing the weights. We have already scaled the weights to protect the channels that have the most salient activations. However, applying naive min-max quantization to these scaled weights is still suboptimal.

Because we scaled up certain channels significantly, the weight matrices now contain extreme outliers. If we simply take the maximum absolute value in a quantization group to determine the grid spacing, those few outliers will force the step size to be massive. The rest of the normal weights will then be crushed to zero or poorly represented, destroying model accuracy.

To fix this, we need to search for an optimal clipping threshold before quantizing. For each quantization group, instead of using `max(abs(w))`, we will evaluate thresholds of the form `c * max(abs(w))` where `c` is a ratio between 0.01 and 1.0. We will scan across a grid of `c` values, quantize the weights using the clipped threshold, dequantize them, and measure the Mean Squared Error (MSE) against the original float weights. We then select the `c` (and its corresponding index) that minimizes the MSE for that group.

Your task is to:
1. Implement the quantization and reconstruction primitive. Use symmetric quantization for a given bit-width (e.g., 4 bits means `-7` to `7`).
2. Implement the grid search to find the optimal clipping ratio `c` per group, returning both the best grid index and the optimal max values.
3. Write a regression test verifying that the search always returns an MSE that is at least as good as not clipping at all (`c=1.0`).
