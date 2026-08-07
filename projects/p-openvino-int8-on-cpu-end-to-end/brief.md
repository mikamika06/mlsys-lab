# Model on CPU under Latency Budget

Our primary inference service is currently deployed on general-purpose cloud instances without dedicated GPU accelerators. The existing PyTorch-based pipeline takes approximately 340 milliseconds per request, which severely violates our strict service-level objective of an 80-millisecond maximum end-to-end latency.

To resolve this bottleneck, we need to transition the inference pipeline to an optimized runtime format, apply post-training INT8 quantization with careful calibration to preserve task accuracy, tune low-level threading and execution hints for CPU architectures, and structure the inference engine to reliably meet the 80 ms latency budget under realistic production workloads.

You are required to implement a modular optimization pipeline that converts the model, profiles execution layers, calibrates and executes INT8 quantization, configures runtime threading parameters, achieves the target latency while maintaining accuracy thresholds, and includes comprehensive regression tests ensuring robustness against performance and correctness regressions.
