# Ticket: Edge Deployment Performance Discrepancies Between ONNX Runtime CoreML EP and Native CoreML

## Symptom
Our mobile engineering team has observed inconsistent latency and execution behavior when deploying computer vision models onto Apple Silicon devices using ONNX Runtime with the CoreML Execution Provider (CoreML EP) versus utilizing native CoreML `.mlpackage` formats directly via coremltools. Specifically, benchmarks on test workloads show unexpected latency spikes and performance degradation on certain model architectures.

Initial investigations indicate that specific operators in the ONNX graph may be falling back from the Apple Neural Engine (ANE) or GPU to the CPU execution path due to lack of direct CoreML operator support, causing costly CPU-ANE context switches and data marshaling overhead. Furthermore, tuning execution options such as forcing `MLComputeUnits=CPUOnly` does not always predictably scale execution profiles across sub-graphs or partitions as expected, leading to regressions in throughput and power efficiency in production environments.

## Goal
We need a robust diagnostic framework and evaluation module within our edge machine learning pipeline (`edgecomp`) to:
1. Compare runtime latency and output consistency between ONNX Runtime CoreML EP and native CoreML `.mlpackage` simulations.
2. Analyze and quantify the performance cost and partition fallback overhead introduced by unsupported ONNX operators.
3. Validate execution options, specifically configuring `MLComputeUnits=CPUOnly`, confirming identical numerical outputs alongside predictable slowdown ratios.
4. Provide comprehensive regression testing to ensure fallback and option configurations remain robust against model modifications.
