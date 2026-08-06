Our automated edge export pipeline targeting LiteRT and TFLite runtimes is currently failing validation across multiple deployment tiers. When pushing our base model through the export harness, downstream edge devices report unexpected runtime validation errors, incorrect buffer types on input/output tensors, and performance instability during quantization calibration sweeps.

Specifically, the export toolchain currently exhibits three distinct failures:
1. The multi-variant generation step fails to correctly produce and size the required four distinct quantization variants (FP32, FP16, Dynamic Range, and Full Integer), leading to incorrect storage footprint estimates and deployment mismatch on resource-constrained hardware.
2. The full-integer conversion routine fails to enforce true int8 input and output tensor bindings, defaulting back to float32 IO wrappers that cause heavy runtime conversion overhead and type mismatch on strict hardware accelerators.
3. The calibration-size stability sweep produces erratic quantization error bounds when varying the size of the representative dataset, lacking a robust stability check to guarantee convergence of activation scales across varying sample counts.

You need to implement the core modules under `edgequant/` and provide a robust regression test suite in `tests/test_regression.py` that guards against regressions in quantization invariants, tensor IO typing, and sweep stability.
