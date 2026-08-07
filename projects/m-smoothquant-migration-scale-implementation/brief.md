# SmoothQuant Migration Scale Implementation & Layer-Wise Alpha Autotuning

Quantizing Transformer activations to INT8 on CPU inference engines causes severe model accuracy drop due to activation outliers. Channel-wise activation quantization is impractical on standard GEMM hardware, so SmoothQuant mitigates this by migrating quantization difficulty from activations to weights via per-channel scaling factors $s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}}$.

When exporting quantized LLMs with Intel Extension for PyTorch (IPEX), applying a static global $\alpha$ across all layers yields suboptimal accuracy. Different layers exhibit distinct outlier profiles: attention projection layers often require higher $\alpha$ values to absorb aggressive activation spikes, while feed-forward networks perform better with lower $\alpha$ values to preserve weight resolution.

## Symptoms

Your automated quantization deployment pipeline is experiencing accuracy regression on CPU target nodes:
- INT8 outputs exhibit high Mean Squared Error (MSE) compared to FP32 reference activations when using fixed static migration factors ($\alpha = 0.5$).
- Downstream tasks fail accuracy validation thresholds across mixed linear layer types.

## Deliverables

1. **`smoothquant/scale.py`**:
   - `compute_migration_scales(act_max, weight_max, alpha)`: Compute per-channel migration scale vector $s$ from maximum absolute activation and weight vector per channel.
   - `apply_smoothquant(activation, weight, scales)`: Scale activation tensor $X$ by $1 / s$ and weight matrix $W$ by $s$ to preserve exact linear output $X \cdot W^T$.

2. **`smoothquant/autotune.py`**:
   - `quantize_int8(tensor, axis=None)`: Symmetric INT8 quantization and dequantization simulator.
   - `sweep_alpha_per_layer(layer_activations, layer_weights, alpha_candidates)`: Perform an alpha sweep across layer candidate values $[0.0, 1.0]$, computing post-quantization output MSE against FP32 reference output to extract optimal per-layer alpha and scales.

3. **`tests/test_regression.py`**:
   - Write unit and regression tests verifying mathematical invariance under scale transformation and ensuring autotuning selects optimal layer-wise migration parameters over static alpha assignments.
