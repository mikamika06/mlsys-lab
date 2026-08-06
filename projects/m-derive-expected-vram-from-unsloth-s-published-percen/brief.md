# Ticket: Validate Unsloth Efficiency Metrics and Benchmarks

We are integrating Unsloth into our automated fine-tuning pipeline to reduce memory usage and increase training throughput across open-weights LLMs. Before rolling out these changes to production clusters, our telemetry team needs precise parsing, expected VRAM derivation, and verified speedup metrics from recorded training logs.

Currently, our benchmark ingestion worker fails to quantify the actual vs. target efficiency gains when processing training logs from different backends. We need a deterministic parser and benchmark analyzer that can compute expected memory footprints based on published reduction claims, extract accurate runtime statistics from unstructured terminal output, and quantify real speedup ratios against vanilla PyTorch / Hugging Face Trainer baselines.

## Deliverables

1. Implement `vram_expected_gb(vanilla_peak_gb, published_pct_savings)` in `unsloth_bench/vram.py` to derive expected VRAM usage given Unsloth percentage-savings claims.
2. Implement `parse_unsloth_log(log_text)` in `unsloth_bench/parser.py` to extract peak VRAM, training steps per second, and final loss from recorded Unsloth stdout console logs.
3. Implement `compute_speedup_ratio(unsloth_steps_per_sec, vanilla_steps_per_sec)` in `unsloth_bench/parser.py` to derive speedup multipliers.
4. Write `tests/test_regression.py` validating that your log parser correctly handles whitespace formatting, float precision, and step-rate calculations across diverse log inputs.
