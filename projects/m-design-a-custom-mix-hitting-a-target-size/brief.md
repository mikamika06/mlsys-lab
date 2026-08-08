Engineers attempting to deploy quantized GGUF models with llama.cpp under strict edge-device memory limits report severe deployment and convergence failures. When automated scripts export quantization recipes for fixed storage budgets, models either exceed the available VRAM capacity or experience severe generation degradation and numerical divergence during token sampling.

An investigation into custom mix recipes revealed two distinct failure modes in the current planning pipeline:

1. 1D tensors (normalization weights, biases, positional scale vectors) are frequently down-quantized alongside 2D weight matrices during global ftype selection. While 2D matrix weights tolerate sub-byte or 4-bit quantizations (such as Q4_K or Q8_0), forcing 1D tensors below full 32-bit floating-point precision (F32) breaks numerical stability during normalization and attention scaling.
2. The model exporter lacks a constrained optimization solver that can select per-tensor or per-layer quantizations under a strict byte size ceiling while guaranteeing that 1D tensors remain invariant at F32 precision across all ftype recipes.

You are tasked with building a quantization recipe planner for GGUF model conversion. You must implement modules that calculate exact tensor storage footprints across mixed-precision formats, enforce and verify the 1D F32 invariant, and select optimal per-tensor quantization formats under hard size budgets.
