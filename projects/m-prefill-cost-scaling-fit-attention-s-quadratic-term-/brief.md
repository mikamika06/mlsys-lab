Long-context models frequently crash in production from unpredicted out-of-memory errors or output garbage due to subtle configuration mismatches. In this unit, we will build tools to predict serving requirements and catch these errors before they cause outages.

You need to build three pieces of functionality:

1. **Prefill Cost Scaling**: The time taken by the prefill phase scales quadratically with context length. Write a function to fit the linear and quadratic coefficients of attention prefill time using historical metrics.
2. **RoPE Diagnostic**: When evaluating a model, you might accidentally combine a checkpoint baked with one RoPE (Rotary Position Embedding) base frequency and a configuration specifying another. Write a diagnostic that detects this mismatch.
3. **Cheapest Serving Config**: To serve an arbitrary context window, calculate the exact memory needed for model weights plus the KV cache. Then, find the cheapest GPU cluster configuration capable of hosting it under a fixed budget.

You'll also write a regression test to ensure that the KV cache size is always factored into memory scaling calculations.
