# m-instrument-a-toy-training-loop-s-4-phases-with-recor

Our local Apple Silicon profiling harness reported inconsistent phase timings for a lightweight PyTorch training loop run on CPU and MPS backends. Visualizing performance logs showed missing annotations around forward, loss, backward, and optimizer steps, while raw event logs contained unmatched trace markers that broke post-processing parsers.

To resolve these discrepancies and establish clean performance baselines, you need to build a lightweight profiling instrumentation module that works without external GPU profiling tools.

First, implement a training loop profiler that wraps the four core execution phases (`forward`, `loss`, `backward`, `optimizer`) using `torch.autograd.profiler.record_function`. The profiler must execute a given training loop under `torch.autograd.profiler.profile` and calculate the relative CPU execution time percentage for each of the four named phases.

Second, construct a trace validator capable of analyzing raw nested event logs (push/pop ranges). The validator must detect unbalanced push/pop calls, returning both an error status and the zero-based index of the first mismatched event tag.

Third, quantify unannotated execution overhead by measuring the percentage of total trace time that occurs outside any named `record_function` scope.

Finally, write a regression test suite in `tests/test_regression.py` that verifies range nesting invariants and correctly identifies mismatched scope markers across profile runs.
