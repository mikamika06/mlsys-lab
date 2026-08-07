We are integrating a new ONNX model, and our nightly latency reports are completely unhinged. The current benchmarking script is heavily flawed. It literally just captures the start time, runs the inference function 100 times in a loop, and divides the total time by 100. This only gives us the mean, which hides the tail latency (like garbage collection pauses or GPU context switching). Even worse, it doesn't warm up the model or the CUDA context before measuring. The first few runs are heavily polluting the numbers because they include memory allocation overhead.

Please write a proper benchmarking methodology in `bench/harness.py`.

First, implement `benchmark(fn, warmup_iters, measure_iters, percentiles)`. It needs to do individual timings per call using `time.perf_counter_ns()` after burning through `warmup_iters` unrecorded calls, and it should return the exact requested percentiles in nanoseconds.

Second, 100 iterations isn't always enough. Implement `find_stable_iters(fn, target_rel_err)` to iteratively find how many runs we need for the 90th percentile estimate to stabilize within the `target_rel_err`.

Finally, add a regression test in `tests/test_regression.py` that verifies the harness actually calls the function the expected number of times, proving that warmup is working.
