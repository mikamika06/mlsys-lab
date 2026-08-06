# QA Evaluation Gate Reporting Inconsistent Quantized Candidate Promotion

Our automated model quantization benchmark runner (`evalrec`) is promoting quantization candidates that cause severe downstream behavioral regressions in production.

During recent model compression runs for 8-bit and 4-bit weight quantization passes, candidate models that passed our post-quantization accuracy checks resulted in unexpected output distribution drift and silent degradation on specialized tasks like math reasoning and code generation. Investigation revealed that candidates were evaluated primarily on aggregate perplexity metrics across all benchmarks, which allowed models with severe tail distribution corruption in specific categories to be promoted.

Additionally, several benchmark evaluation runs reported suspiciously low perplexity scores due to contaminated or corrupted FP16 reference baselines, yet the evaluation suite flagged no warnings and approved the quantized candidates against these tainted targets.

We need a structured accuracy evaluation and recovery gate package (`evalrec`) that:
1. Ranks quantization candidates using per-category KL divergence against the reference FP16 baseline alongside perplexity, flagging cases where perplexity and KL divergence rankings disagree.
2. Identifies contaminated or abnormal reference baseline distributions (such as sudden perplexity drops or collapsed distribution entropy).
3. Enforces a strict accept/reject gate using explicit per-category thresholds for both KL divergence and perplexity increase before approving any quantized candidate.
