# Layerwise Isolation and Divergence Analysis in Polygraphy

During model export and precision calibration pipelines (ONNX to TensorRT), numerical mismatches frequently arise between the ONNX Runtime reference execution and TensorRT engine outputs. Isolating the exact point of divergence across deep network graphs requires systematic, layer-by-layer comparison of intermediate activation tensors.

Currently, engineers spend excessive time manually writing verbose Polygraphy CLI command invocations (`polygraphy run`), executing them against intermediate subgraphs, and running custom diff scripts to pinpoint divergent nodes. When models scale to hundreds of layers, this manual process becomes error-prone and unmaintainable.

You need to build a programmatic toolset that automates Polygraphy's layer-wise isolation and numerical analysis pipeline. Your package must:
1. Construct exact CLI commands that trigger Polygraphy layer-wise execution and mark intermediate outputs for dumping.
2. Parse comparative activation outputs to identify the first divergent layer under relative and absolute tolerance thresholds.
3. Implement core error statistical metrics (mean absolute error, relative error, signal-to-noise ratio, and maximum absolute difference) to reproduce Polygraphy's internal comparison report.
4. Provide a regression test suite that validates tolerance thresholding and catches false positives when divergence criteria are mutated.
