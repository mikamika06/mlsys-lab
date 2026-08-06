We are observing significant memory bottlenecks in our multi-head attention service when processing extended context lengths under standard FP16 KV caching. To relieve memory bandwidth and footprint constraints, we plan to adopt per-head FP8 KV cache quantization ($e4m3fn$). However, team members have raised concerns regarding how quantizing KV tensors impacts attention quality, specifically whether context scaling introduces catastrophic attention matrix error at long sequence positions.

Your task is to implement a light FP8 KV quantization pipeline, compute real memory footprint and context capacity gains across various hardware configurations, implement per-head FP8 affine quantization/dequantization, and measure attention accuracy loss across token positions.

To address this, you will:
1. Build memory allocation estimators that compare FP16 vs FP8 per-head KV cache byte requirements and context length capacity given a GPU memory budget.
2. Implement per-head FP8 scale quantization and dequantization routines for Key and Value tensors, ensuring proper per-head axis scaling.
3. Compute attention error (relative error of attention outputs) across token sequence positions comparing FP16 reference attention against quantized FP8 attention, and write regression tests to ensure improper scaling or quantization routines are caught.
