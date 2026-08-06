# Failure Triage: Isolate Quantization, Tokenizer, and Engine Errors

Our production LLM serving pipeline for low-bit GGUF models (`llama.cpp` ecosystem) has begun reporting erratic degradation across diverse client workloads. Some deployments output gibberish text or repeated sequences, others fail downstream schema parsing due to malformed tokenization, and a third group produces numerically degraded logit distributions or early termination sequences.

Because our automated deployment pipeline combines quantization, custom tokenizer configs, and low-level runtime engines, field engineering cannot easily pinpoint which component is responsible when a failure occurs.

Your task is to build a root-cause isolation and log triage tool to systematically differentiate between:
1. **Tokenizer damage**: Incorrect token ID mapping, vocabulary offset corruption, or broken special token handling (e.g., `BOS`/`EOS` handling).
2. **Quantization damage**: Excessive perplexity spikes, NaN/Inf tensor values, or severe logit drift beyond standard low-bit rounding tolerance.
3. **Engine execution failure**: Out-of-bounds context window indexing, CUDA/CPU buffer alignment errors, or kernel execution panics.

## Objectives
- Implement `isolate_root_cause(sample)` to analyze diagnostic metrics and classify the single root cause among tokenizer damage, quantization damage, or engine execution failure.
- Implement `triage_log_batch(samples)` to process batches of failure logs and achieve >= 95% accuracy against verified reference diagnostics.
- Write automated tests in `tests/test_regression.py` that verify your diagnostic isolation logic and detect regressions when failure thresholds or classification boundaries are violated.
