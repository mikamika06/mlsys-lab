# 2:4 Sparsity That Gave No Speedup

We recently applied 2:4 fine-grained structured sparsity to our Transformer linear layers, pruning two out of every four contiguous weight elements along the inner K-dimension. Although exactly 50% of the weight parameters are now zero in our model checkpoint, our end-to-end inference benchmark shows zero latency reduction. The wall-clock execution speed is identical (1.0x) compared to the original unpruned dense baseline.

The performance team needs an engineering diagnostic suite to investigate why structured sparsity fails to accelerate this workload:

1. Validate whether the weight matrices strictly conform to the 2:4 fine-grained sparsity pattern required by hardware sparse Tensor Cores.
2. Trace the runtime GEMM dispatcher to identify which execution path was selected and diagnose any fallback triggers (e.g., misaligned matrix dimensions or small batch sizes).
3. Measure operational metrics (FLOPs and memory access) on native 2:4 sparse Tensor Core kernels versus dense fallbacks.
4. Construct a roofline performance model to calculate the arithmetic intensity break-even threshold where 2:4 structured sparsity transitions from memory-bound to compute-bound speedups.
5. Produce a quantitative performance report proving whether speedup is physically possible for the target shape and hardware configuration, or demonstrate speedup above the break-even limit.
6. Confirm that the 2:4 compressed sparse checkpoint achieves a smaller storage footprint on disk compared to the uncompressed dense model.
