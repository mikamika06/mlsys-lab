# Ticket: Model OOMs during fine-tuning under memory constraints

## Symptom
Our distributed fine-tuning job fails with CUDA Out-Of-Memory (OOM) during the first optimizer step when using standard ZeRO configurations. Engineers have been blindly toggling ZeRO stages, CPU offload, and per-device batch sizes until the run stops crashing. However, this trial-and-error approach has caused severe throughput drops and unexplained network bottlenecks across cluster nodes.

## Goal
We need an accurate memory and communication planner for ZeRO-powered training. You must implement a estimator module that predicts:
1. Exact GPU memory consumption (weights, gradients, optimizer states, activations) across ZeRO-1, ZeRO-2, and ZeRO-3.
2. Total inter-GPU communication traffic per step for each ZeRO stage.
3. Step latency overhead introduced by CPU offloading.
4. Optimal stage and batch size selection under a strict memory budget, accurately predicting memory within a ±15% tolerance.
5. Verification of predictions against recorded execution traces.
6. Scaling behavior predictions when doubling the GPU count.

Fix this by building a principled analytical model in `zero_planner` rather than guessing parameters.
