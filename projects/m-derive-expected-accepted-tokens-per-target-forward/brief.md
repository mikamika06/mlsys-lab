SYMPTOM:

When deploying speculative decoding with draft models in our llama.cpp serving pipeline, production benchmarks report highly erratic throughput numbers across hardware platforms. On server-class GPU nodes, running speculative decoding with `--draft-max 8` yields a 1.8x acceleration over target-only generation. However, when deploying the exact same configuration to CPU edge instances, inference generation throughput drops by over 25% compared to running the target model alone without speculative drafting.

Field logs show that operators currently select the `--draft-max` hyperparameter using static heuristics or trial-and-error manual tuning on isolated benchmark prompts. When token acceptance rates degrade due to out-of-domain prompt distributions or heavy draft step execution overhead, the extra forward passes for draft verification become a performance bottleneck rather than an acceleration mechanism.

We need an automated utility module within our serving stack that analyzes speculative verification traces, computes the expected accepted token yields, models generation throughput under specific draft and target step latencies, and dynamically selects the optimal draft maximum length.
