We've recently started porting some of our fine-tuning workloads from a pure NVIDIA GPU cluster to a hybrid setup that includes Apple Silicon (MPS). To keep memory usage down on the MPS nodes, we added `torch.autocast("mps", dtype=torch.float16)` since some of our operations didn't support `bfloat16` perfectly on older macOS versions.

However, we are now experiencing `inf` and `NaN` values in the loss almost immediately. We suspect two things:
1. Is `autocast` actually casting the intermediate activations, or did we misconfigure it? Are the weights being safely left in `float32` as intended?
2. We think a dot product in the attention layer is overflowing the `float16` maximum representable value (which is around 65504), a limit we didn't have to worry about with `bfloat16` (max ~3.4e38).

We need a diagnostic tool to settle this. Your task:
- Implement `inspect_autocast(model, x, device_type, autocast_dtype)` to run a forward pass under autocast, capturing and returning the `dtype` of the model's output, the `dtype`s of all leaf modules' intermediate activations (using forward hooks), and the current `dtype` of all leaf modules' `.weight` parameters.
- Implement `synthesize_overflow()` to return two 1D `torch.Tensor` objects (float32) such that their element-wise product sum overflows to `inf` when cast to `float16`, but computes safely without overflowing when cast to `bfloat16`.
- Provide regression tests to ensure the synthesized tensors actually overflow `float16` but survive `bfloat16`.
