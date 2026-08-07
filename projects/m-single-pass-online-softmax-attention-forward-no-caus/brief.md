# Single-Pass Online-Softmax Attention Forward & Causal Masking

We are adding Triton-based kernel components to our internal low-level kernel library (`fused_attn`). Our downstream service currently relies on standard PyTorch attention calls during training and evaluation, which incurs significant memory overhead from allocating $O(N^2)$ intermediate attention matrices and performing separate scaling, exponentiation, and reduction passes.

Your task is to build a numerically stable, single-pass fused online-softmax attention forward kernel and extend it to handle causal sequence masking. In addition, you must implement diagnostic profiling tools that calculate the theoretical floating-point operations (FLOPs) of the attention forward pass and derive actual TFLOPS performance from recorded wall-clock execution times.

## Symptoms & Observed Limitations
1. Memory usage spikes quadratically with sequence length during full attention forward passes because full attention maps are materialized in HBM.
2. Naive multi-pass implementations require multiple reads and writes of intermediate softmax scores to global memory, limiting kernel throughput.
3. Adding causal masking dynamically by applying explicit attention masks causes unnecessary arithmetic operations on masked-out upper-triangular blocks.

To resolve these issues, you need to implement online running maximum and online scale update mechanics across key/value blocks, split block iterations efficiently between off-diagonal and diagonal tiles for causal masking, and verify performance tracking using exact FLOP derivations.
