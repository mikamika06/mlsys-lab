# Ticket: Gradient Clipping and Loss Scaling Anomaly in Mixed-Precision Training

## Symptom Report
Our distributed and single-node mixed-precision training runs using PyTorch's `GradScaler` are exhibiting erratic gradient norms and unexpected divergence when gradient clipping is enabled alongside dynamic loss scaling. Specifically, models trained with fp16 and `GradScaler` show periods where effective gradient clipping appears completely ineffective during scale factor adjustments, while bf16 baseline runs without a scaler do not suffer from this discrepancy.

Additionally, monitoring dashboards report incorrect counts of skipped optimizer steps when underflow conditions occur. When gradients underflow and the scaler skips the optimizer step, our internal tracking metrics continue to increment the global step counter or fail to log the skipped status accurately, skewing our learning rate schedulers and loss tracking metrics.

We need to isolate the exact mechanism causing these discrepancies, correct the execution order within our training step utility, properly account for skipped optimizer steps under underflow workloads, and add a rigorous regression test suite to ensure that any future regressions in gradient unscaling and clipping sequence are caught automatically before merging into the main branch.
