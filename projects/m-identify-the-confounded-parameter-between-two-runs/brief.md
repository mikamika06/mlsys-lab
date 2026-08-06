# Ticket: Confounded Parameters and Throughput Scaling in llama-bench

We are investigating performance regression anomalies observed when benchmarking Llama models using `llama-bench` within `llama.cpp`. During a recent series of profiling runs, engineers noticed that prompt processing (pp) throughput behavior diverges unexpectedly when altering thread counts and batch sizes simultaneously.

Specifically, when comparing two distinct benchmark runs on the same hardware with identical model weights, Run A and Run B exhibit contradictory throughput scaling trends. Preliminary analysis suggests that one or more parameters—likely related to thread allocation (`-t`), batch size (`-b`), or physical batch chunking (`-ub`)—are heavily confounded, masking the true hardware saturation point and invalidating the performance comparison.

Furthermore, token generation (tg) throughput patterns do not consistently align with expected memory bandwidth utilization or sequential bytes-read orderings across different thread configurations. This lack of monotonic scaling makes it difficult to optimize execution flags for maximum throughput.

Your task is to analyze the runtime parameter interactions, isolate the confounded parameter causing the divergence between the two benchmark configurations, determine the optimal configuration tuning for `-t`, `-b`, and `-ub` to surpass default pp throughput, and implement verification routines to check that tg ordering strictly matches expected bytes-read orderings. Complete the required reference and test modules to pass all validation milestones successfully.
