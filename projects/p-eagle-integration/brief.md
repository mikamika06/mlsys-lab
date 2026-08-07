# Ticket: Eagle Draft Head Integration

Separate draft models used in speculative decoding introduce significant memory overhead because they duplicate large embedding tables, attention layers, and transformer block parameters alongside the primary target model. This leads to high VRAM consumption and non-trivial inter-model communication or orchestration complexity during generation.

A more efficient alternative is to attach a lightweight draft head directly onto the intermediate hidden states or final layers of the target model. This design reuses the target model's massive feature extraction backbone, requiring only a tiny fraction of additional parameters and memory.

However, integrating a draft head introduces specific systems challenges:
1. **Hidden State Alignment**: Correctly extracting and passing the target model hidden representations into the draft head without breaking forward passes or KV-cache management.
2. **Speculative Sampling & Verification**: Ensuring that tree-based or sequential draft token generation matches the target distribution semantics during verification rounds.
3. **Acceptance Rate & Speedup**: Measuring effective token acceptance ratios to confirm that the overhead reduction translates into genuine computational efficiency.
4. **Memory Footprint Verification**: Quantifying the VRAM savings compared to running a standalone separate draft model.
5. **Temperature Robustness**: Maintaining stable speculative generation and acceptance behavior under varying sampling temperatures.

Your task is to implement the draft head modules, wire them into the inference engine, verify correctness across workloads, ensure substantial memory savings, validate speedup thresholds, and protect the system against regression under temperature changes.
