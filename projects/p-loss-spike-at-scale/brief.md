# Ticket: Loss spikes at 64 GPUs

We are experiencing severe instability when scaling our distributed training job. When running on 8 workers, the model training converges beautifully and the loss decreases monotonically. However, when we scale the cluster out to 64 workers, the loss periodically spikes to massive values or goes completely to NaN, completely ruining the optimizer state.

The spikes typically appear right around step 45 on our test benchmark. We initially suspected bad data or a corrupted batch, but we've thoroughly verified our dataloaders. Running the exact same batch of data globally doesn't cause this anomalous behavior, which strongly points to a subtle numerical issue rather than a data pipeline issue.

Our core suspicion is that something in our distributed aggregation mechanism—likely a global sum or a gradient norm computation—is accumulating floating-point errors as the number of active workers increases in the cluster. At 64 workers, the accumulated precision error seems to overflow or lose significant digits catastrophically.

Your task is to investigate this systematically and isolate the fault:
1. Build a log analyzer to programmatically find exactly when the spike happens.
2. Prove that the data sharding itself mathematically preserves the invariant.
3. Write a test to check if our reduction operations are order-independent (commutative).
4. Implement a safe, numerically stable version of our all-reduce operation.
5. Validate that a training step using your new reduction matches the expected exact math and stays stable.
6. Finally, write a regression test that fails if someone accidentally reverts to a precision-lossy reduction.
