# Quantify the Super-Scale Amortization Advantage

We are converting model weights to k-quant block formats (specifically super-blocks like `Q4_K` / `Q5_K` / `Q6_K` used in `ggml` / `llama.cpp`). During profiling, we noticed our quantization and scale-factor calculation routines have overheads that scale non-linearly with block layout parameters.

A naive block quantization scheme uses independent scale factors for small sub-blocks, leading to high metadata overhead and frequent scale calculation loops. Super-block quantization groups multiple sub-blocks (e.g., 256 elements in 8 sub-blocks of 32 elements) to amortize high-precision scale and min parameters across the entire super-block while keeping sub-block scales low-bit.

We need a clean python module `kquant/amortization.py` that quantifies the metadata overhead, byte footprint, and effective quantization error across uniform and super-block quantized layouts. You will also provide a safety regression suite in `tests/test_regression.py` that verifies scale amortization and invariant behavior across arbitrary block configurations.
