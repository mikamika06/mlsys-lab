# Quantization Precision Regression in Low-Bit MXFP4 Decoding

Production inference pipelines using gguf models report occasional numerical instability and severe quantization noise when evaluating layers compressed with MXFP4 (Microscaling FP4) encoding. During model import, floating-point weights are block-quantized with a shared exponent, but downstream kernel evaluations suffer from inaccurate reconstructions and misaligned error profiles compared to legacy integer formats like Q4_0.

Engineers observed that decoding FP4 sub-elements with shared E8M0 scale factors produces higher-than-expected absolute error across specific dynamic ranges. To pinpoint where precision breaks down, we need to implement and validate an isolated MXFP4 decoding pipeline, analyze the complete spectrum of representable FP4 grid values across varying scale exponents, and quantify the exact error crossover threshold where MXFP4 loses fidelity against standard 4-bit uniform block quantization.

You will implement the block decoding logic, build a grid enumerator for representable MXFP4 values, and construct an automated regression test suite to detect numerical drift and broken scale-factor scaling.
