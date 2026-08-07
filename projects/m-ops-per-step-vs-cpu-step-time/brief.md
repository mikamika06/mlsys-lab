# Ops per step vs CPU step time

We are observing severe throughput degradation when running small-batch workloads on our PyTorch inference pipeline. Although the total FLOPs per step scales down as expected with reduced batch sizes, the overall step execution time plateaued far earlier than predicted. Profiles indicate that GPU compute utilization drops dramatically while CPU-side overhead remains constant, leaving the hardware severely underutilized.

To resolve this issue, we need a formal analytical framework to model the transition between launch-bound and compute-bound regimes. You must implement tools to analyze operation launch overheads against GPU execution times, identify critical kernel launch thresholds, and predict potential speedups under small-batch regimes.

Your task is to build analysis functions that measure CPU overhead versus GPU execution bounds, determine the minimal operation count needed to maintain GPU saturation above 80%, and forecast speedups when transitioning small-batch workloads out of launch-bound limits.
