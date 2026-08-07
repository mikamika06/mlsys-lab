# Edge Weight Export Quantization: Symmetric vs Affine Scale & Zero-Point Derivation

## Symptom & System Issue
When exporting large transformer language models to Core ML for Apple Neural Engine execution (`coremltools.optimize.coreml`), low-bit quantization causes dramatic perplexity degradation. An automated export pipeline was configured to quantize weight tensors to 4-bit integers using naive symmetric scaling. However, profile outputs show severe truncation error across asymmetric activation-projection layers where weights exhibit significant mean shift.

Further inspection reveals that whole-tensor symmetric quantization forces zero-point symmetry around zero, stretching the scale parameter and wasting dynamic range. Attempts to mitigate this by switching to affine (scale + zero-point) quantization or blockwise quantization ran into broken size calculations and incorrect relative error metrics ($rel\_err$).

You need to implement an edge quantization analysis and transformation library. This includes deriving symmetric and affine quantization parameters, computing blockwise 4-bit compressed storage size ratios on real layer configurations, and conducting a block-size error sweep to find optimal operating points on accuracy versus memory size.

## Requirements
You will write functions in three modules:
1. `qcore/derive.py`: Derive exact scale and zero-point parameters for both symmetric and affine 4-bit quantization schemes.
2. `qcore/stats.py`: Perform blockwise INT4 quantization/dequantization and calculate precise byte size ratios between uncompressed FP16 weights and INT4 blockwise representations (including scale and zero-point metadata overhead).
3. `qcore/sweep.py`: Measure $rel\_err$ (relative $L_2$ error) and perform block-size sweeps to evaluate quantization error curves.
4. `tests/test_regression.py`: Implement regression tests ensuring that symmetric zero-points stay zero, blockwise size ratios properly account for metadata overhead, and smaller block sizes achieve lower relative error.
