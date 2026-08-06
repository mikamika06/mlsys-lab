We recently started routing some of our heavy NLP models through the TensorRT execution provider in ONNX Runtime. While the end-to-end latency clearly improved in our dashboards, my team is having a hard time explaining exactly *where* the time went, and whether our target operator fusions (like specific `MatMul` and `LayerNormalization` nodes) are actually doing the heavy lifting.

When I dump the ORT profiles and load them into `chrome://tracing`, it’s an absolute mess. There are thousands of node events, and the first few inference runs are painfully slow—a classic warmup phase—which completely skews any simple averages I try to calculate manually. 

We need a robust tool that can programmatically sift through these JSON trace dumps. First, it should find the point where the warmup ends and the steady-state begins. Second, I want to identify the 80% hot-op cover set: the smallest set of operator types that account for at least 80% of the steady-state node execution time. Finally, we need to compare a "before" and "after" trace to attribute the exact time saved (or lost) per operator type on average per run.

Please implement the analyzer module. We run this in CI, so make sure to add a regression test ensuring our hot-op cover threshold calculation is perfectly accurate and doesn't terminate a node early.
