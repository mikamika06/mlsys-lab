The post-training quantization pass is injecting massive noise into our vision model. We expected standard int8 precision drops, but the final activation logits are completely mangled.

When I dumped the activation distributions, the errors after quant/dequant round trips are sometimes exceeding 1.5x the expected bin width. I strongly suspect it's because our activations, which pass through ReLUs and are therefore strictly non-negative, are being forced into a symmetric quantization scheme. This wastes half the representable range. We need asymmetric quantization primitives that track a zero-point to accurately encode skewed distributions.

Furthermore, the convolution weights are currently quantized per-tensor. Some channels have huge outliers that absolutely destroy the resolution for the rest of the tensor. We must migrate the weight scales to be per-channel instead.

Finally, the C++ runtime team requires the exact "fused requantization scale" to map the int32 convolution accumulator directly to the next layer's int8 input. If they do floating point math dynamically for `(acc * weight_scale * input_scale) / output_scale` it's too slow; they need the precomputed float scalar so they can derive the integer multiplier and shift.

Please build these primitives and mathematically verify the error bounds so we can salvage this integer inference pipeline.
