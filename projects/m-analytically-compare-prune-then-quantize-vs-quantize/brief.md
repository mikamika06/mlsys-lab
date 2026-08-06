# Ticket: Order Effects in Joint Pruning and Quantization

A downstream service team reports that applying model compression using a naive pipeline yields lower accuracy than expected when combining weight pruning and quantization. Specifically, when compressing linear layers, applying quantization before pruning results in significant MSE reconstruction loss on activation outputs compared to pruning before quantization, or using a joint SparseGPT-style reconstruction. 

The compression pipeline needs an analytical estimation tool, a joint transformation execution module, and a test suite to prevent order-inversion regressions during automated model updates.

### Task Description

You need to implement the order-effect analysis and joint compression functions in `order_effect/`:

1. **Analytic Comparison (`order_effect/analytic.py`)**:
   Implement `compare_order_error(W, X, sparsity, num_bits)` to compute the analytical reconstruction mean squared error (MSE) between dense model output $Y = W X$ and compressed outputs:
   - $Y_{\text{prune\_then\_quant}}$: Magnitude-prune $W$ to `sparsity` (zero out smallest absolute magnitude weights per row), then uniformly affine quantize non-zero weights to `num_bits`.
   - $Y_{\text{quant\_then\_prune}}$: Uniformly affine quantize $W$ to `num_bits`, then magnitude-prune the quantized tensor to `sparsity`.

2. **Joint SparseGPT Pipeline (`order_effect/pipeline.py`)**:
   Implement `run_joint_compression(W, X, sparsity, num_bits, block_size=64)` to run a SparseGPT-style joint pruning and quantization step:
   - Compute the inverse Hessian proxy $H^{-1} = (X X^T + \lambda I)^{-1}$ where $\lambda = 10^{-4} \cdot \text{trace}(X X^T) / d_{\text{in}}$.
   - Process columns in blocks of `block_size`. For each column, decide whether to prune based on weight magnitude updated by prior Hessian corrections, and uniformly quantize surviving weights while compensating remaining uncompressed weights in the block using Hessian update steps.

3. **Regression Tests (`tests/test_regression.py`)**:
   Provide tests verifying that order effects match analytical bounds and that joint SparseGPT compression achieves lower reconstruction MSE than isolated quantize-then-prune pipelines across dynamic activation inputs.
