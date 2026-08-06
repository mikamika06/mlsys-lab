# AWQ Per-Channel Scale Search on a Linear Layer

Production activation-aware weight quantization (AWQ) relies on searching for optimal per-channel activation scales $s$ to protect salient weight channels before quantization. The core principle leverages a grid search over grid parameters $\alpha \in [0, 1]$ where per-channel scale candidates are derived from activation magnitude bounds: $s = s_X^\alpha$. To apply these scales without overhead during inference, we must mathematically fold scale factors into weight matrices and activation paths without altering the linear layer's underlying theoretical output.

Recently, our quantization pipeline's per-channel scale grid search has been producing suboptimal mean squared error (MSE) reconstructions on quantized linear layers. Upstream engineering reports that quantized layer outputs show non-deterministic error spikes and incorrect scale selections during AWQ optimization, suggesting either numerical distortion in the scale-folding invariant or flawed grid searching across channel scale bounds.

Your task is to fix and implement the AWQ per-channel scale search pipeline:
1. Prove and implement exact, function-preserving scale folding across activation scale transformations and linear weight updates.
2. Implement an empirical search across the $\alpha \in [0, 1]$ grid, sweeping candidate scaling vectors $s = s_X^\alpha$, measuring output reconstruction MSE between FP16 and INT4-quantized linear outputs, and selecting the optimal $\alpha$ argmin.
3. Construct a regression test suite in `tests/test_regression.py` that verifies scale-folding exactness and catches invalid scale folding where activation scaling is wrongly decoupled from weight rescaling.
