# Symptom: Severe Latency Spikes and Priority Inversion under High Concurrency

Our high-throughput LLM serving infrastructure is experiencing unacceptable latency tail-spikes during peak traffic hours. Telemetry from the serving layer shows that when concurrency surges, high-priority interactive requests are stuck behind long-running background batch requests.

Furthermore, increasing request concurrency past a critical saturation threshold causes a sharp drop in overall system throughput rather than maintaining peak processing capacity. Engineers suspect the issue lies in how iteration-level scheduling and KV block allocation interact under constrained memory and token budgets.

We need an iteration-level scheduler simulator modeling vLLM's internal scheduling loops, KV block management, and token budgeting. You will implement the discrete simulator, evaluate system throughput across a range of concurrency levels, and build automated regression testing that proves priority scheduling resolves priority inversion when high-priority requests arrive under heavy background load.
