# Accuracy evaluation and recovery reporting for quantized LLMs

A post-training quantization pipeline for a 7B language model was deployed across several low-bit precision variants (FP8, INT4, INT2). The evaluation team reported inconsistent model behavior across quantization formats: while task-level accuracies appeared stable on some benchmark suites, downstream text generation degraded with noticeable token repeating and early EOS termination.

Existing validation tools only log top-1 task accuracy from benchmark harnesses, hiding fine-grained probability distribution shifts and token-level prediction flips. Furthermore, recovery metrics are currently computed manually across ad-hoc JSON outputs, leading to conflicting reporting across evaluation runs.

You need to build a lightweight evaluation utility that:
1. Calculates token-level Kullback-Leibler (KL) divergence and top-1 token flip rates from paired baseline and quantized logit arrays.
2. Evaluates sequence perplexity directly from stored log-probabilities for arbitrary sequences.
3. Parses standard `lm-evaluation-harness` result JSON structures to compute normalized recovery percentages and generate structured quality summary reports.
4. Includes a suite of invariant regression tests to ensure metric stability against corrupted or degenerate logit distributions.
