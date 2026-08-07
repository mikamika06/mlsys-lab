# Deployment OOMs and Unexpected CPU Fallback during LLM Quantization Selection

When deploying LLM inference instances across heterogeneous GPU nodes (Tesla T4, RTX 4090, Apple M2 Ultra), our auto-selection pipelines regularly trigger Out-Of-Memory (OOM) failures or exhibit severe 10x throughput degradation.

Investigation revealed two root causes in the selection logic:
1. On older hardware architectures like Tesla T4 (CUDA compute capability 7.5) or Metal 2.0, non-linear codebook matrix-multiplication kernels for IQ quant types (e.g., IQ3_XXS, IQ2_XS) are unsupported on GPU. When selected, these models silently fall back to CPU memory or allocate unexpectedly large intermediate staging buffers.
2. Quantization size estimation logic failed to account for non-weight overheads (such as KV cache and activation scratch buffers), causing higher-precision quants to be placed on memory-constrained GPUs where total allocation exceeded hardware VRAM limits.

We need a lightweight module `quantplan` containing:
- `quantplan.backend.will_fallback_to_cpu`: predicts whether an IQ quantization type will fall back to CPU execution on a given backend configuration.
- `quantplan.picker.estimate_vram_bytes`: computes model weight size plus overhead bytes.
- `quantplan.picker.find_best_quant_index`: determines the optimal candidate quantization index (`argmin_index` ordered by perplexity) that satisfies memory budget and GPU kernel execution requirements.
