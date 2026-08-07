Local runner deployments frequently hit Out-Of-Memory (OOM) crashes or experience severe latency degradation despite initial static VRAM calculations matching model weight sizes. System telemetry indicates that operator memory planning miscalculates total resident VRAM by treating weight quantization as uniform across layers and ignoring the combined footprint of weight blocks, key-value (KV) caches, and dynamic activation buffers during decode.

You need to implement an accurate memory footprint predictor and bandwidth simulator for local LLM inference runners. The tool must parse heterogeneous model configurations, calculate exact byte footprints across weights, KV caches, and activation buffers, and predict decode throughput based on memory bandwidth constraints.

Specifically, you will build modules that:
1. Compute total resident VRAM by accounting for non-uniform K-quantization block layouts across attention, feed-forward, and normalization layers, alongside multi-head KV cache allocations and runtime activation bounds.
2. Explain layer-wise allocation trade-offs by evaluating why mixed K-quant precision (e.g., higher precision for attention and gate projections vs lower precision for down projections) preserves key representation quality within a fixed VRAM budget.
3. Predict decode generation speed in tokens per second as a function of total memory movement per token and effective system bandwidth.
4. Author regression tests to ensure memory footprint and throughput bounds hold under config changes and detect regressions such as uniform weight bit-width assumptions or missing activation overheads.
