# Allocated-Bytes Timeline and Leak Diagnosis from PyTorch Snapshots

We are diagnosing an unexpected out-of-memory (OOM) error occurring during the execution of a custom PyTorch training loop on CUDA. Although the theoretical model footprint (weights, gradients, and optimizer states) should consume comfortably less than the available memory, GPU memory usage steadily rises until an OOM crash occurs mid-epoch.

A PyTorch CUDA memory snapshot (`torch.cuda.memory._snapshot()`) was dumped right before the system crashed. However, raw memory snapshot data contains unstructured trace segments, raw frame IDs, allocation events, and frame stacks that are difficult to analyze manually.

Your task is to build a snapshot analyzer tool in `snaptool/` that parses the memory snapshot trace to accomplish three tasks:

1. **Reconstruct the Allocated-Bytes Timeline:** Parse snapshot allocation/free history events to construct a timeline of total active allocated bytes over time, and compute peak allocation statistics.
2. **Identify Retaining Frames:** Trace un-freed block allocations back through their call stacks to identify the exact frame (function name and line number) responsible for holding retained memory across steps.
3. **Compare Theoretical Footprint vs. Peak Memory:** Calculate the expected static memory footprint from model parameters and compare it against the snapshot's observed peak memory to isolate dynamic overhead and leaks.

Finally, write regression tests in `tests/test_regression.py` that verify your diagnostic metrics correctly identify leaking stack frames and accurately measure theoretical vs peak memory allocations.
