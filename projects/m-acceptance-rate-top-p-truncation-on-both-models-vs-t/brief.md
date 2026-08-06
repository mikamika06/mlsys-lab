# Acceptance Rate Degradation under Dual Top-P Truncation

## Symptom

Our speculative decoding engine exhibits a severe drop in token acceptance rate when nucleus sampling ($top\text{-}p$) is enabled on both the draft model and the target model simultaneously. During standard speculative execution under temperature sampling, acceptance rates strictly follow the expected theoretical bounds for probability distribution matching. However, when both models independently apply top-$p$ truncation prior to sampling and verification, accepted draft lengths plunge significantly below baseline expectations, causing unnecessary draft rejections and high decoding latency.

## Diagnosis Needed

We need to analyze and implement precise distribution matching for speculative sampling when top-$p$ truncation is applied across both models versus target-only truncation.

Specifically:
1. Implement the formal acceptance/rejection criterion for speculative decoding when top-$p$ nucleus truncation is applied to both target and draft distributions versus when applied to the target distribution alone.
2. Measure the theoretical and empirical acceptance rate across varying temperature ($T$) and top-$p$ thresholds to quantify the distribution shift introduced by dual-model truncation.
3. Construct a regression test suite that verifies distribution fidelity and catches incorrect acceptance probability calculations, such as misapplying draft-side top-$p$ masks to the target distribution during the residual sampling step.

Fix the distribution matching logic and acceptance check in `spec/sampling.py` and `spec/acceptance.py`, and add regression tests in `tests/test_regression.py`.
