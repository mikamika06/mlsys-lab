# Symptom: Student distillation loss flattens unexpectedly early with high-temperature teachers

When distilling smaller student models from over-parameterized teacher outputs, the training loss encounters an unexpectedly high loss floor despite sufficient capacity for the downstream task. Diagnostics show that applying standard Knowledge Distillation (KD) soft targets with high temperatures ($T > 2.0$) produces a systematic mismatch between the student's expressible logit space and the teacher's target distribution.

Furthermore, short KD runs regularly report sudden drops in prediction entropy on low-margin classes, leading to localized mode collapse where the student outputs uniform distribution spikes for distinct input clusters. Increasing temperature fails to resolve this collapse and instead shifts the learned logits away from the teacher's true soft margin.

You must build diagnostic tools to analyze and correct these distillation failure modes:

1. **Synthetic Capacity Gap Simulation**: Compute theoretical and empirical loss floors when a low-rank or lower-capacity student attempts to match a high-capacity teacher distribution across varying temperature scales.
2. **Mode Collapse & Entropy Diagnostic**: Detect early-stage mode collapse in KD runs by tracking entropy decay across class slices and evaluating effective temperature shifts when teachers display overconfidence.
3. **Regression Prevention**: Implement test assertions in `tests/test_regression.py` that catch subtle miscalibrations in effective temperature adjustments and detect premature entropy collapse under teacher overconfidence.
