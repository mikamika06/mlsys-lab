Our high-throughput compilation pipeline relies on manual TensorIR schedule transformations to optimize matrix multiplication kernels. Recently, downstream deployment targets have reported numerical mismatches and performance regressions on dense GEMM workloads.

Initial triage reports that loop transformation pipelines are breaking tensor invariants or producing sub-optimal throughput. Specifically, engineers observed:
1. Intermediate loop transformation steps occasionally introduce correctness bugs due to mismatched loop bounds or incorrect axis reordering.
2. The deployed kernels are failing to achieve target speedups over naive loop nests, with suspected failures in loop vectorization and thread-level parallelization.
3. Analysis scripts that inspect TVM TensorIR string representations fail to accurately trace which loop axes correspond to which schedule transformations.

To resolve these failures, you need to build a structured scheduling and evaluation utility in `tirsched/`. You will construct a naive TIR matrix multiplication PrimFunc, programmatically apply loop split, reorder, vectorize, and parallel transformations, verify numerical correctness against a reference baseline after each step, and map axis annotations from printed TIR loop nests back to their schedule transforms. Finally, you will measure the speedup of the scheduled kernel relative to the naive version, and write a test suite in `tests/test_regression.py` that verifies the safety of the transformation pipeline when schedule primitives are altered or omitted.
