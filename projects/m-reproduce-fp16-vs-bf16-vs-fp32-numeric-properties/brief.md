# Diagnostic Report: Unexpected Mixed-Precision Training Instability and Loss Divergence

## System Description
Our model training pipeline leverages PyTorch `torch.cuda.amp.autocast` to execute forward passes in lower-precision floating-point formats (`fp16` and `bf16`). The goal is to accelerate computation and reduce GPU memory footprint while maintaining numerical stability and final evaluation accuracy comparable to full-precision FP32 training.

## Symptom & Observed Behavior
During standard pre-training runs, switching the autocast precision mode from IEEE half-precision (`fp16`) to Bfloat16 (`bf16`) or standard single-precision (`fp32`) results in severe numerical divergence, unexpected activation dynamic range clipping, and subtle underflow issues:

1. **Underflow / Quantization Step Artifacts in FP16/BF16:** When computing standard numerical statistics (such as dynamic range, subnormal limits, epsilon, and smallest positive non-zero denorm/norm values), model activations experience severe precision degradation. Small gradient updates vanish prematurely under `fp16` due to its coarse mantissa precision, while underflow limits differ sharply between `fp16` and `bf16`.
2. **Inconsistent Op Autocasting Behavior:** Operators such as matrix multiplications (`torch.matmul`, `nn.Linear`), convolutions, softmax, layer normalization, cross-entropy loss, and simple additions exhibit non-uniform casting policies. Certain normalization and loss functions are unexpectedly retained in `fp32` by `autocast`, while linear projections are cast to lower precision. Without a accurate catalog of operator behavior, custom activation functions and loss wrappers end up silently running in low precision, causing catastrophic numerical overflow.
3. **Activation Overflow Anomalies in FP16:** Assuming activations across transformer blocks follow an approximate Gaussian distribution $\mathcal{N}(0, \sigma^2)$ after normalization, activations exceeding $65504.0$ trigger `NaN` or `Inf` loss spikes. We lack an exact theoretical model for the theoretical overflow tail probability $P(|X| > 65504)$ as a function of the activation standard deviation $\sigma$.

## Objective
Implement a core precision diagnostic and evaluation module in `numprec/` that:
- Accurately computes exact numerical format properties (mantissa bits, exponent bits, min/max positive values, subnormal limits, machine epsilon, and relative representation errors) for `fp16`, `bf16`, and `fp32`.
- Classifies operator casting policies under PyTorch `autocast` for key Deep Learning primitives.
- Provides a theoretical overflow probability calculator for FP16 activations under Gaussian statistics, alongside a comprehensive regression test suite to catch critical numerical degradation bugs.
