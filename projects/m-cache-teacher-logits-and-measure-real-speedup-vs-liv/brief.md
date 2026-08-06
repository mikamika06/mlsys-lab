Our distillation pipeline for training compact student models from a 7B teacher model is exhibiting severe throughput bottlenecks during multi-epoch distillation runs. Despite the student model having significantly fewer parameters and lower FLOP requirements than the teacher, total epoch wall-clock times remain nearly identical to full teacher evaluation per step. Profiling indicates that for every batch step, the training loop synchronously executes a full forward pass through the teacher model to generate target logit distributions before taking a student optimizer step.

When team members attempted an unmanaged pre-computation script to cache teacher output logits onto host RAM and disk, the cluster nodes repeatedly crashed with Out-Of-Memory (OOM) errors during the cache generation phase for datasets exceeding 10 million tokens.

We need a dedicated distillation cache module to:
1. Exactingly calculate host RAM and memory footprint requirements for full-vocabulary vs top-k compressed teacher logit storage across custom vocabulary sizes and sequence lengths.
2. Implement a teacher logit caching engine that eliminates redundant teacher forward evaluations across epochs.
3. Quantify the net throughput speedup and wall-clock overhead reduction achieved by offline logit caching compared to online live-teacher distillation.
