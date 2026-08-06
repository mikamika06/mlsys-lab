# Real Speedup & Storage Validation Decision Table Across Sparsity

## Symptom
Our deployment engineering team has received complaints from model serving teams regarding unfulfilled performance and storage compression targets. Developers exported several pruned PyTorch checkpoints (`.pt`) using standard `torch.save()`, expecting immediate file size reductions proportional to the layer sparsity. However, disk storage metrics show zero reduction in size for `.pt` files, and memory allocation remains identical to dense baselines.

Simultaneously, inference microbenchmarks for 2:4 (N:M) structured sparse GEMMs reveal a persistent speedup gap on Tensor Cores: theoretical operations per second suggest up to a 2.0x performance boost, but actual hardware measurements consistently underperform expectations or degrade into negative speedups for non-aligned shape configurations.

## Goal
You must construct a verified speedup and storage decision engine that validates low-level sparsity claims against actual GPU runtime and storage constraints.

1. **Storage & Sparse Overhead Analysis**: Implement functions to calculate exact theoretical bit-level sizes for dense, CSR, COO, and 2:4 structured sparse formats. Prove why saving a dense tensor containing zero values in PyTorch (`.pt`) yields no file size reduction, and compute the critical sparsity threshold where sparse representation overhead offsets memory savings.
2. **Tensor Core N:M Constraint & Speedup Gap**: Implement exact matrix shape validation for Ampere/Hopper Tensor Core 2:4 sparse GEMM hardware (16-byte alignment, tile dimensions, $K$-dimension multiples of 32/64). Calculate true achievable speedup accounting for index metadata packing, memory bandwidth limits, and execution tile efficiency.
3. **Decision Table Engine & Safeguard**: Build a decision table generator that recommends whether to convert a candidate linear layer to dense, 2:4 N:M structured sparse, or fine-grained unstructured CSR/COO based on target speedup, bandwidth bounds, and shape constraints. Finally, write a suite of regression tests in `tests/test_regression.py` that verifies that your decision engine correctly rejects illegal Tensor Core tile alignments and prevents unviable sparse conversions.
