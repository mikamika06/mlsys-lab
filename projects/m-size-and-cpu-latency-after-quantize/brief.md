# Ticket: CPU memory footprint and latency anomalies after quantization

## Reported Symptom
During our latest deployment cycle for CPU-targeted inference, our automated benchmark suite reported that running `quantize_` on transformer modules did not yield the expected ~4x reduction in memory footprint. On several submodules, memory consumption remained identical to FP32 baselines, and CPU matrix multiplication latency benchmarks showed no measurable improvement.

Additionally, our model inspection and debugging pipeline—which needs to reconstruct full-precision FP32 weight matrices directly from stored torchao INT4 `state_dict` artifacts—is failing. Ingested state dicts with packed INT4 weight tensors and associated scale/zero-point parameters are either raising shape mismatches or returning corrupted dequantized float values when decoded.

## Impact
We cannot verify whether quantization actually modified tensor storage layouts in place or silently performed a no-op due to unsupported tensor shapes or dtypes. Furthermore, offline model analysis and fallback FP32 validation are blocked because we lack a verified weight recovery routine for INT4 state dict formats.

## Goal
Implement diagnostic and recovery utilities to:
1. Benchmark model weight byte sizes and CPU execution latency before and after quantization, computing exact size ratios and performance deltas.
2. Interrogate model parameters to diagnose no-op `quantize_` calls where weight tensors were left unquantized or unchanged.
3. Unpack and dequantize torchao-style INT4 state dict parameters back into original FP32 float weight tensors.
4. Add regression tests to ensure dequantization and diagnostic checks catch broken or corrupted state dict layouts.
