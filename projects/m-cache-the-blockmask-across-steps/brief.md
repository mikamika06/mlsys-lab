# Ticket: Autoregressive Generation Stalls During Long Sequence Inference

## Symptom Report
During autoregressive text generation with FlexAttention at sequence lengths exceeding 4096 tokens, generation latency spikes drastically with every single token appended. Profiling the generation loop reveals that the primary bottleneck is not the attention computation itself, but rather the repeated instantiation and recompilation overhead occurring inside the step loop. Specifically, the dynamic creation of attention masks and their associated block sparsity structures are being re-evaluated from scratch at every generation step, even when the prefix tokens and their valid causal boundaries remain completely static. 

Furthermore, engineers inspecting intermediate verification metrics have noticed two critical anomalies during validation runs:
1. When comparing block sparsity against element-level sparsity configurations, memory footprint and dispatch overhead diverge unexpectedly from theoretical expectations, causing inefficient kernel launches.
2. In certain custom attention patterns where query and key-value index arguments were refactored, the underlying spatial coordinate mapping in the mask modifier behaves inversely, leading to severe attention leakage across diagonal block boundaries.

We need to optimize our incremental decoding loop by persisting and reusing the cached attention structures, properly quantifying block versus element sparsity benefits, and correcting coordinate mapping errors in the custom mask functions without breaking existing regression invariants.
