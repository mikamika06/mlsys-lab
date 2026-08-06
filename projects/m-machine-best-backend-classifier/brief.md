# Auto-backend selection and platform failure classification for FlashAttention

## Symptom
Production LLM serving instances running FlashAttention show unpredictable latency spikes and runtime crashes across heterogeneous GPU nodes. On modern Hopper architecture (SM90), workers occasionally crash with cryptic CUDA errors when trying to dispatch unsupported attention kernels. On older Ampere nodes, the engine frequently selects sub-optimal legacy backends, leading to high fallback overheads and memory pressure.

To stabilize serving across mixed clusters, we need an automated machine classifier that inspects target hardware capabilities and dynamic tensor properties to select the optimal FlashAttention backend. When a platform constraint or capability check fails, the system must explain the exact root cause rather than silently crashing. Additionally, we need to quantify the precise latency and memory cost of falling back from specialized hardware backends (like FlashAttention-3 or Triton-optimized routines) to generic PyTorch attention paths.

## Requirements
1. Implement a backend classification engine in `fa_backend/classifier.py` that maps hardware capability tuples (Compute Capability major/minor, SM count, SRAM per SM, Tensor Core support) and input tensor dimensions (head dimension, data type, causality, variable sequence lengths) to the optimal execution backend (e.g., `FA3_HOPPER`, `FA2_TRITON`, `FA2_CUDA`, or `MATH_FALLBACK`).
2. Implement platform failure diagnosis functions in `fa_backend/failure.py` that evaluate why a target machine cannot run a given backend, returning explicit failure reasons (e.g., insufficient compute capability, misaligned head dimensions, lack of FP8/FP16 Tensor Core support, or dynamic shared memory exhaustion).
3. Implement a fallback cost estimator in `fa_backend/cost.py` that calculates memory bandwidth ratios, FLOPS overhead, and estimated latency degradation when falling back to standard math or baseline implementations.
4. Implement regression tests in `tests/test_regression.py` validating classifier correctness and failure explanations under perturbed machine configurations.
