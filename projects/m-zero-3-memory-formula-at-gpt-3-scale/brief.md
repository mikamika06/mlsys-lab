Our distributed training pipeline is hitting OOM errors when scaling to GPT-3 sizes using ZeRO Stage 3 (full parameter sharding). The cluster operators want a dry-run tool to predict memory footprints, communication volumes, and schedule overlaps before jobs are submitted.

Your task is to build a memory simulator for ZeRO-3:

1. **Memory Math**: Implement `zero3_memory_math` to compute exact memory metrics. You must compute:
   - `sharded_bytes`: 16 bytes per parameter (fp16 params, fp16 grads, 12-byte fp32 Adam states) evenly sharded across GPUs (use `//`).
   - `baseline_peak_active_bytes`: Peak fp16 parameter footprint assuming JIT gathering with no prefetch (just the largest layer).
   - `comm_per_gpu_bytes`: The 3*Psi total communication volume (fp16 all-gather fwd, all-gather bwd, reduce-scatter bwd) per GPU.

2. **JIT Schedule**: Implement `build_schedule` to generate the sequence of events (`all_gather_fw`, `compute_fw`, `free_fw`, `all_gather_bw`, `compute_bw`, `reduce_scatter`, `free_bw`), supporting a lookahead `prefetch` window. Then build `simulate_peak_memory` to track the maximum bytes of fully-materialized fp16 layers held simultaneously over the course of the schedule.

3. **Safety Net**: Write `test_compute_requires_active_memory` in `tests/test_regression.py` to ensure that whenever a layer is computed or reduced, it is actively materialized in memory, protecting the JIT pipeline against aggressive premature freeing bugs.
