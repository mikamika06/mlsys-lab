We received reports from downstream pipeline engineers that our custom Triton-style block vector processing kernel produces silent numerical errors and memory corruption when processing non-standard input sizes.

When input tensor sizes are exact multiples of the tile block size (such as 64, 128, 256, or 1024), all downstream outputs, checksums, and regression checks pass without issue. However, whenever model inference runs on variable sequence lengths or odd batch shapes where tensor length N is not divisible by the tile size, numerical drift and unmasked writes occur in subsequent pipeline layers.

Investigation shows that existing unit tests were only executed on standard block-aligned dimensions (e.g., 512, 1024, 2048). As a result, the existing test harness marked the kernel as fully functional despite hidden array out-of-bounds reads and unmasked stores occurring during tail block processing.

Your task is to reproduce the corruption on non-block-aligned input sizes, identify the corrupted index boundaries beyond N, implement proper load/store tail masking in the kernel execution logic, sweep boundary sequence lengths, verify byte-for-byte correctness against reference outputs across 50 arbitrary sizes, and supply regression tests that fail if improper tail masking is re-introduced.
