# Fine-Tuning Framework Performance Profiling Anomalies

## Problem Statement
Our internal cluster evaluation pipeline ingests benchmark run logs from three fine-tuning frameworks across various multi-GPU and single-GPU configurations. Downstream scheduling decisions and framework recommendations are currently failing because performance comparison metrics are inaccurate or inconsistent.

Specifically, operators report that speedup metrics comparing frameworks do not isolate matched model configurations, producing distorted speed ratios. Furthermore, framework efficiency rankings mistakenly favor frameworks with higher memory consumption, and multi-GPU scaling efficiency metrics report impossible linear scaling (> 100%) because single-GPU baselines are miscalibrated.

## Requirements
You need to implement structured comparison utilities in `benchcomp/ratios.py` and `benchcomp/scaling.py`:

1. In `benchcomp/ratios.py`:
   - Implement `compute_pairwise_ratios(records)`: Group records by identical `(config_id, num_gpus)` tuples. For every pair of frameworks `(fw_a, fw_b)` present in a configuration, compute the throughput ratio `tokens_per_sec(fw_a) / tokens_per_sec(fw_b)`. Return a dictionary mapping `(fw_a, fw_b)` to the arithmetic mean ratio across all matched configurations.
   - Implement `rank_frameworks(records)`: Calculate mean throughput (`tokens_per_sec`) and mean peak VRAM usage (`vram_gb`) per framework across all benchmark records. Return a dictionary `{"speed": [...], "vram": [...]}` where `"speed"` lists framework names sorted descending by throughput, and `"vram"` lists framework names sorted ascending by peak VRAM footprint (lower memory usage ranked higher).

2. In `benchcomp/scaling.py`:
   - Implement `compute_scaling_efficiency(records)`: Match multi-GPU records (`num_gpus > 1`) with their 1-GPU baseline for the same `(framework, config_id)`. Compute scaling efficiency as `tokens_per_sec(N_gpus) / (tokens_per_sec(1_gpu) * N_gpus)`. Return a dictionary mapping `(framework, num_gpus)` to the arithmetic mean efficiency across available configurations.

3. In `tests/test_regression.py`:
   - Write unit tests covering `compute_pairwise_ratios`, `rank_frameworks`, and `compute_scaling_efficiency`. The tests must verify that VRAM ranking places lower memory usage first and that scaling efficiency correctly accounts for GPU count normalization.
