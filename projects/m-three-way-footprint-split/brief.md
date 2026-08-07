# Edge Model Footprint Analyzer & Selective Build Tool

We are preparing to deploy a suite of vision and text models to resource-constrained edge devices with tight memory ceilings. During pre-flight profiling, several edge runtimes failed to boot or crashed with Out-Of-Memory (OOM) errors during early inference passes.

Field diagnostics indicate three distinct root causes:
1. Binary size and initial memory overheads are being lumped together, making it difficult to isolate static executable bloat from initial runtime heap allocations.
2. Unused operator kernels are being compiled into the engine binary, unnecessarily inflating the executable footprint.
3. Peak Resident Set Size (RSS) during multi-tensor inference exceeds the available RAM ceiling because temporary activation allocations and workspace buffers are not modeled accurately prior to execution.

We need a systematic tool to dissect the footprint, prune unused operator registrations, and predict peak memory consumption before flashing firmware to physical targets.

Your task is to build the footprint analysis library in `footprint/`:

1. In `footprint/split.py`, implement `analyze_three_way_footprint()` to separate memory into a three-way split: static binary footprint, runtime infrastructure overhead, and dynamic tensor allocation footprint.
2. In `footprint/selective.py`, implement `selective_registration_win()` to calculate binary and operational footprint savings achieved by stripping unregistered kernels based on model operational graphs.
3. In `footprint/predictor.py`, implement `predict_peak_rss()` to estimate peak RSS by simulating sequential tensor lifetimes, alignment padding, and allocator overheads across inference graphs.
4. In `tests/test_regression.py`, write unit tests that validate footprint accounting invariants, verify selective registration savings bounds, and ensure peak RSS predictions fail properly when activation lifetimes overlap incorrectly.
