# Ticket: KV Cache Quantization Inaccuracies under Low Precision FP8 Format

**Symptom:**
During dynamic KV cache quantization, our serving runtime exhibits silent accuracy degradation when serializing key-value states into sub-byte or compressed formats. Debugging traces indicate that current float-to-int/float-to-fp8 conversion logic introduces bitwise discrepancies when target representations (specifically FP8 E4M3) undergo float32 casting, scaling, and rounding. In addition, when comparing standard E4M3 against E5M2 representations across active attention key/value dumps, large dynamic range shifts lead to unexpected mean squared error (MSE) spikes, leading to corrupt token generation.

**Task:**
You need to implement a bit-exact FP8 (E4M3) bit-manipulation encoder and decoder in NumPy, compute per-tensor optimal scaling factors from `absmax` values, and run comparative precision error analysis (E4M3 vs E5M2 MSE metrics) using cached KV tensor dumps. Finally, write a unit test suite to guard against dynamic range dynamic scaling regressions.
