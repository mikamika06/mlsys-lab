# Multi-LoRA Memory Budgeting and Batch Scheduling

Production deployments of vLLM serving hundreds of customized LoRA adapters are crashing under unexpected out-of-memory (OOM) errors during peak traffic spikes. While base model weights and Key-Value (KV) cache memory usage are strictly accounted for, multi-adapter serving introduces memory overheads that fluctuate based on active adapter weights and request batching strategies.

Operations teams report two distinct issues:
1. Serving nodes run out of GPU memory when loading multiple user adapters simultaneously, even when total batch size remains constant.
2. Under heavy traffic with mixed adapter requests, inference throughput degrades severely due to frequent adapter swap operations and unoptimized batch groupings.

To restore serving stability and high throughput, we need a transparent framework for adapter memory accounting and adapter-aware request scheduling.

## System Requirements

You must implement a three-part module under `loraserve/`:

1. **Adapter Size Calculator (`loraserve/config.py`)**:
   Parse HuggingFace PEFT `adapter_config.json` parameter dictionaries alongside base model weight shapes to compute exact adapter memory footprints (in bytes) across target modules ($W_A$ and $W_B$ weight matrices).

2. **Server-Side Memory Budgeter (`loraserve/budget.py`)**:
   Calculate static memory preallocations required for multi-LoRA serving clusters based on configured `max_loras` and `max_lora_rank` slots, and determine whether incoming adapter sets fit within strict memory caps.

3. **Adapter-Aware Batch Scheduler (`loraserve/scheduler.py`)**:
   Schedule incoming inference requests into batches that maximize adapter reuse across steps, minimizing costly adapter switches while staying within batch slot and memory limitations.

Finally, write regression tests in `tests/test_regression.py` to ensure your scheduling logic correctly distinguishes requests requiring adapter switches from contiguous single-adapter batches.
