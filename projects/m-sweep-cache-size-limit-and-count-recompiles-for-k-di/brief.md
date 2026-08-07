# Dynamo Guard Storm and Cache Limit Exhaustion

We are observing severe latency spikes in our PyTorch 2.0 compiler deployment when handling dynamic workloads with diverse input shapes. During traffic bursts, the compilation phase repeatedly triggers recompile storms, causing excessive memory consumption and unpredictable fallback to eager execution.

Engineering leadership suspects that `torch.compile` cache limits and guard specializations are interacting poorly with varying tensor dimensions. When the compiler receives inputs exceeding the shape specialization cache limit, it silently stops compiling and forces eager fallback, but we currently lack fine-grained telemetry to detect when and why this occurs.

To fix this issue, you must implement a diagnostic and guard-tracking framework for `torch.compile`. Your module must:
1. Track recompile counts across K distinct input shapes under varying `cache_size_limit` settings and detect when eager fallback is triggered.
2. Parse guard logs (equivalent to `TORCH_LOGS=guards` output) to extract exact shape-guard expressions and identified guarded dimensions.
3. Simulate recompile storms to calculate cache exhaustion thresholds and track exact fallback transitions.
4. Provide a regression test suite that verifies that guard exhaustion and eager fallback events are correctly caught when cache size limits are exceeded.
