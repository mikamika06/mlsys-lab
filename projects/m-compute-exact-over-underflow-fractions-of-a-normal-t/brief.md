# Ticket: Numerical Stability and Quantization Errors in Training Diagnostics

During recent high-throughput mixed-precision pre-training runs, several model training instances exhibited unusual loss progression, sudden divergence, and gradient degradation. Diagnostic logs indicate multiple distinct failure modes, including unexpected zero-gradients, NaN losses, loss plateaus, and loss spikes. Additionally, there are concerns that repeating standard INT4 (NF4) quantization-dequantization cycles across intermediate checkpoint saves or low-bit activation passes is accumulating silent numerical drift.

To address these issues, we need an automated analysis library that can evaluate numerical behavior under reduced precision formats (FP16 vs BF16), accurately classify training log anomalies into their root causes, and simulate numerical decay across multi-step NF4 quantization cycles.

## Task Requirements

1. **Precision Boundary Analysis (`numdiag/overflow.py`)**:
   Implement `compute_overflow_underflow_fractions(tensor, dtype)`. Given a floating-point NumPy tensor, calculate the exact fraction of elements that result in overflow (greater than the maximum representable finite value or negative infinity) and underflow (non-zero values that become subnormal or flush to zero) when converted to FP16 and BF16 formats.

2. **Log Anomaly Classification (`numdiag/classifier.py`)**:
   Implement `classify_training_log_symptoms(log_entry)`. Analyze structured training log entries containing loss metrics, gradient norms, and learning rate telemetry, and map them to one of four root numerical failure modes: `FP16_UNDERFLOW`, `FP16_OVERFLOW`, `GRADIENT_VANISHING`, or `REPRESENTATION_COLLAPSE`.

3. **NF4 Quantization Error Simulation (`numdiag/quantization.py`)**:
   Implement `simulate_nf4_compounding_error(tensor, num_cycles)`. Simulate sequential iterations of NormalFloat4 (NF4) quantization followed by dequantization on an input tensor. Calculate relative error statistics across iterations to track error accumulation.

4. **Regression Safeguards (`tests/test_regression.py`)**:
   Implement test cases in `tests/test_regression.py` that verify your diagnostic logic against numerical stability edge cases and ensure that classifier thresholds reliably catch invalid precision assignments.
