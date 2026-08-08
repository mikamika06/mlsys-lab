Team,

We are spending too much time manually correlating metrics across three different profilers (Nsight Compute, our custom Proton profiler, and PyTorch's built-in profiler). We need to automate the extraction and classification of key performance metrics from the raw outputs of these tools.

I've outlined three tasks to get us started:

1. **NCU Trace Classification**: Write a function `classify_ncu(metrics)` that takes a list of dictionaries with `name`, `sm_pct`, `mem_pct`, and `warps_pct`.
   - If `sm_pct >= 80.0`, return `"compute_bound"`.
   - Else if `mem_pct >= 80.0`, return `"memory_bound"`.
   - Else if `warps_pct < 60.0`, return `"occupancy_bound"`.
   - Otherwise, return `"latency_bound"`.
   Return a dictionary mapping the kernel name to its classification.

2. **Proton Trace Exclusive Time**: Write `analyze_proton(events)` that takes a trace of `"enter"` and `"exit"` events (dictionaries with `time`, `type`, and `region`). Calculate the **exclusive time** for each region as a percentage (`0.0` to `100.0`) of the total trace duration. Total trace duration is the difference between the maximum and minimum `time` across all events. Return a dictionary mapping region name to this percentage.

3. **PyTorch Trace TFLOPS Estimate**: Write `analyze_torch(events, flops_per_thread)` to parse a list of PyTorch trace JSON dictionaries. Filter for `cat == "kernel"`. Using the `Grid X/Y/Z` and `Block X/Y/Z` dimensions in the `args` dictionary (defaulting missing dimensions to 1) and the kernel's `dur` in microseconds, compute the average TFLOPS across all invocations of each kernel. `flops_per_thread` maps the kernel name to FLOPs per thread. Return a dictionary mapping the kernel name to its average TFLOPS.

Lastly, add a safeguard regression test to ensure our Proton parser invariant (that exclusive time percentages for a gapless nested trace sum to exactly 100%) isn't accidentally broken in the future.

Thanks,
ML Systems Team
