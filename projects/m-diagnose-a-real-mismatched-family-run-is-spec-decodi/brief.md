# Ticket: Speculative Decoding Run Showing Degradation on Production Workload

## Symptom Description
Our recent deployment of local speculative decoding using a smaller draft model alongside our target production model has raised significant performance concerns. While initial offline benchmarks suggested a speedup when using a draft model from a related family, our live trace telemetry indicates that end-to-end token generation latency has actually increased compared to running the target model alone.

Users are reporting sluggish responses and higher time-to-first-token variance during peak traffic intervals. Examining the raw execution logs reveals that despite the draft model generating candidate tokens at high raw throughput, the target model's verification pass is rejecting the vast majority of these proposals. Specifically, consecutive accepted token lengths rarely exceed one or two tokens, and frequently drop to zero when handling domain-specific prompts or out-of-distribution inputs.

Furthermore, the overhead incurred by running the draft forward pass, computing speculative sampling distributions, and managing KV cache synchronization across mismatched architectures appears to outweigh any throughput gains. We need a robust diagnostic utility integrated into our evaluation suite to accurately analyze trace logs, compute precise acceptance rates, determine the theoretical breakeven threshold, and definitively diagnose whether a given speculative decoding configuration is net-helping or net-harming before promoting it to production environments.
