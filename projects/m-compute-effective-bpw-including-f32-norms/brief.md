We are running a cluster of small VMs with varying VRAM constraints. When loading GGUF models, the `Q4_K_M` or `Q8_0` labels suggest a specific bits-per-weight (bpw), but a model's true VRAM footprint almost always exceeds this naive calculation. The 1D tensors—primarily layer norms and RMS norms—are preserved in F32 (32 bits-per-weight) in GGUF to maintain numerical stability, pushing the actual bpw higher than the nominal quant size.

In this project, you will track model memory accurately:
1. Implement `compute_effective_bpw` to calculate the true bits-per-weight, treating all 1D tensors as 32 bpw and other tensors as the base quant bpw.
2. Implement `select_quantization` to choose the highest quality (largest) quantization that fits into a strict `vram_budget_bytes`. Remember that the VRAM budget must also accommodate the KV cache (which uses FP16, meaning 2 bytes per element, with keys and values for every layer, and every token up to the context length).
3. Calculate the `size_ratio` (the ratio of quantized weight bytes to standard FP16 weight bytes) for each quantization option.
4. Write a safeguard test that would fail if someone incorrectly applied the base quant bpw to the F32 norms.
