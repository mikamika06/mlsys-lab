# Incident Report: OOM Failure in MoE Long-Context Deployment under Combined Offload Flags

## Symptom
During long-context inference runs using mixture-of-experts (MoE) models on limited VRAM hardware via `llama.cpp`, jobs intermittently fail with an out-of-memory (OOM) error. This specifically happens when combining expert offloading to CPU with aggressive context-window settings and tensor parallelism or flash-attention flags. The process terminates abruptly during the prefill phase or early decoding steps, well before the theoretical maximum context length defined by the model configuration is reached.

## Investigation Context
When scaling context length for sparse MoE models, the key-value (KV) cache memory scales linearly with sequence length, while expert weight tensors occupy a massive static VRAM footprint. To mitigate VRAM exhaustion, operators often enable expert offloading to CPU. However, interactions between CPU-offloaded expert routing, dynamic batch sizing, and KV cache allocation can lead to unexpected memory spikes, fragmentation, or incorrect tracking of available VRAM overhead. We need a robust implementation to calculate the exact VRAM and CPU memory requirements for expert-offload versus full-GPU expert placement, diagnose the exact combined-flag OOM triggers in `llama.cpp`, and write automated regression tests to prevent recurrence.
