# Ticket: Inconsistent Inference Outputs After Optimum Intel Model Export

## Symptom
During CPU deployment tests, models quantized using the low-level `nncf.quantize()` pipeline show minor numerical divergence when compared to models produced by the one-shot `OVQuantizer.quantize()` interface. The downstream team reported that while both pipelines target OpenVINO INT8 execution, the intermediate graph representations generated during export retain different ONNX opset metadata. This discrepancy leads to different fallback behaviors during graph optimization on x86 targets, causing activation divergence across batch runs.

## Expected Behavior
The one-shot `OVQuantizer` workflow and the manual `nncf.quantize` pipeline must yield bit-identical quantization parameters and output activations within an acceptable relative error tolerance (`rel_err <= 1e-4`). Furthermore, intermediate opset metadata must be systematically identified and harmonized before applying NNCF calibration and quantization transformations.

## Tasks
1. Implement intermediate ONNX opset version detection from graph configurations and model nodes to guarantee operational alignment.
2. Build the manual `nncf.quantize` pipeline and align its quantization parameters with `OVQuantizer` one-shot quantization to achieve output parity.
3. Add a safeguard regression test suite in `tests/test_regression.py` to ensure broken quantization target parameterizations are immediately flagged.
