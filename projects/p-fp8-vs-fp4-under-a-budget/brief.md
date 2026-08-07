# Project Ticket: Choosing Sub-Byte Quantization Formats Under a Strict Memory Budget

Modern large language models require substantial device memory to hold weights during inference, often exceeding single-accelerator capacity. When deploying models under a strict memory budget, system engineers must choose between low-bit floating-point formats such as FP8 (E4M3 / E5M2) and ultra-low-bit sub-byte formats like FP4, which pack multiple values into a single storage word and incur varying block-structure overheads.

Directly applying these formats leads to complex trade-offs involving effective bits-per-weight (bpw) storage overhead, quantization error distribution, end-to-end task degradation, hardware kernel instruction support, and architectural constraints across disparate accelerator generations.

Your task is to implement an analysis framework that evaluates, measures, and selects the optimal low-bit quantization configuration under a hard memory constraint. You must compute precise effective bpw including metadata block headers, measure quantization noise on reference tensors, evaluate end-to-end task performance, analyze kernel execution compatibility, produce a data-backed recommendation, and ensure your workflow generalizes correctly across multiple simulated hardware targets.
