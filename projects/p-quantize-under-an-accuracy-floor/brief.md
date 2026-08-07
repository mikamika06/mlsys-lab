# Ticket: Quantize Under an Accuracy Floor

We need to halve the memory footprint of our deployed model while preserving at least 99% of its baseline evaluation metric (losing at most 1.0% accuracy on our evaluation suite).

Currently, full-precision models are straining our inference memory footprint and bandwidth. Moving blindly to uniform 4-bit or 8-bit quantization often drops performance below acceptable production thresholds or causes catastrophic failure on sensitive attention/projection layers.

Your goal is to build an end-to-end, reproducible quantization pipeline with high-fidelity evaluation and selective mixed-precision capabilities:

1. **Build a reliable evaluation baseline:** Establish a deterministic benchmark harness that measures model accuracy across standard evaluation tasks.
2. **Calibration dataset engineering:** Analyze the impact of calibration sample size and representative dataset distribution on post-training quantization scales.
3. **Compare quantization schemes:** Evaluate baseline recipes (uniform INT8, uniform INT4) under identical calibration conditions.
4. **Identify sensitive layers:** Measure layer-wise error sensitivity and construct a mixed-precision configuration that keeps critical layers at higher precision (e.g., INT8) while quantizing the majority to lower precision (e.g., INT4).
5. **Meet size and accuracy constraints:** Produce a final quantized candidate that meets the budget (>= 2x compression) while staying within the accuracy floor requirement (<= 1% accuracy drop).
6. **Ensure test suite regression defense:** Provide robust regression tests in `tests/test_regression.py` that can detect sensitivity calculation failures and improper mixed-precision fallback logic.
