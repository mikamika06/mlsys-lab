# Tensor Parallelism Mechanics: Column/Row Parallel Matmul, Communication Volume, and DTensor MLP

During scale testing of our distributed Megatron-style Transformer stack on an 8-GPU mesh, multi-node training runs crashed or produced non-deterministic outputs. Downstream monitoring flagged two distinct failures: activation outputs from the MLP block diverged from single-device ground truth when scaled across world sizes, and network link bandwidth saturated prematurely despite having sufficient memory for weights.

A initial post-mortem indicates that tensor parallel primitives lack clean rank-aware partitioning mechanics. Specifically, matrix multiplications in the feed-forward layer are not handling forward identity and backward reduction semantics (`f` and `g` autograd operators) correctly. Furthermore, our cluster autotuner cannot balance rank communication overhead because we lack an analytical formula estimating per-layer tensor parallel communication volume under varying TP worlds and batch configurations.

To address these issues, you will build and test the core tensor-parallel building blocks:
1. **Column and Row Parallel Matmul From Scratch:** Implement standard Megatron-LM `f` and `g` autograd functions with forward column/row split matrix operations and backward all-reduce / identity gradient handling using PyTorch distributed collectives.
2. **Per-Layer TP Communication Volume Formula:** Write an analytical communication volume estimator function for standard Megatron column-parallel and row-parallel layers under specified batch, sequence, hidden, and tensor parallel size parameters.
3. **Real DTensor Column/Row-Parallel MLP:** Implement an MLP module leveraging PyTorch DTensor mesh layout concepts (`Shard` and `Replicate` placements) that processes distributed tensors across ranks and generates bitwise-accurate hidden representations matching sequential reference layers.

Finally, write an automated test suite in `tests/test_regression.py` that catches gradient and activation divergence across tensor parallel ranks when `f` or `g` autograd operators misplace identity and all-reduce passes.
