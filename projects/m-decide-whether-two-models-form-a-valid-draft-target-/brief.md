# Speculative Decoding Validation and Performance Diagnostics

We are attempting to deploy speculative decoding to accelerate token generation in our llama.cpp/GGUF serving stack. A team member paired several candidate draft models with a target model, but the serving pipeline started throwing runtime errors, crashing on specialized prompts, or slowing down significantly during initial prompt processing.

When inspecting the cluster logs, three primary operational symptoms surfaced:
1. Model loading crashes or silent corruption occur when initializing speculative decoding sessions. Inspection indicates vocabulary mismatch, divergent special token configurations (such as BOS, EOS, or padding IDs), or mismatching GGUF tokenization metadata between draft and target models.
2. In several production benchmarks, speculative generation runs slower than standalone target generation. The team has no analytical tool to evaluate whether a draft model's acceptance rate exceeds the theoretical break-even threshold given the draft-to-target execution cost ratio and draft step count.
3. Attempts to apply speculative speculative draft steps during the prompt processing (prefill) phase cause latency spikes. There is currently no measurement utility to compute prefill execution latency overhead and demonstrate why speculative decoding fails to accelerate prompt processing.

Your goal is to build a validation and performance diagnostic module in `specdec/` that verifies draft/target model pair compatibility, calculates the exact analytical break-even acceptance rate, and quantifies prompt processing overheads. Finally, write unit tests in `tests/test_regression.py` that guard against pairing incompatible models.
