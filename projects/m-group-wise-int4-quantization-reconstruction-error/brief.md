# Group-wise Int4 Quantization Reconstruction Error

During CPU inference optimization for large language models, our quantization pipeline experienced severe accuracy degradation when switching from uniform per-tensor int8 quantization to group-wise int4 quantization. Downstream evaluation showed high mean squared error (MSE) on weight reconstruction, especially on layers with outlier weight distributions.

Upon inspection, the current group-wise int4 quantizer produces incorrect scale factors, handles zero-point alignments inconsistently across smaller group sizes, and misclassifies saturation-clipping elements. When weights exceed the int4 grid bounds $[-8, 7]$, the clipping logic either truncates prematurely or fails to count saturated vs inside-bound elements correctly, leading to distorted reconstruction error calculations.

Your task is to fix the group-wise int4 quantization module and build an error analysis pipeline.

You need to:
1. Implement symmetric/asymmetric group-wise int4 quantization and reconstruction in `quant/group_quant.py`.
2. Compute accurate mean squared error (MSE) reconstruction metrics and classify weight elements into saturated (clipped) versus non-saturated elements across dynamic group sizes in `quant/metrics.py`.
3. Add a suite of regression tests in `tests/test_regression.py` that catches flawed saturation classification and incorrect group scale calculations.
