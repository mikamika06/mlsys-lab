## Symptom

When profiling large language model inference workloads on Apple Silicon using tools like `asitop`, engineers frequently notice discrepancies between the summarized metrics displayed on screen and the granular values recorded in raw `powermetrics` exports. Furthermore, diagnosing pipeline stalls requires analyzing exported Metal System Trace logs to identify command-buffer scheduling gaps, as well as detecting sudden mid-run utilization drops during long generation loops.

## Mission

Build the `edgemetrics` profiling toolkit to parse raw system telemetry, cross-check tool summaries, analyze command-buffer gaps, and detect residency drops programmatically.

## Milestones

1. **Powermetrics Parser & Cross-Check**: Implement `parse_powermetrics` and `cross_check` in `edgemetrics/parser.py` to accurately extract GPU and CPU power metrics from raw outputs and verify them against displayed summaries.
2. **Metal System Trace Gap Counter**: Implement `count_gaps` in `edgemetrics/trace.py` to scan exported trace events and count command-buffer scheduling gaps exceeding specified duration thresholds.
3. **Residency Logger & Safeguard**: Implement `detect_drop` in `edgemetrics/logger.py` to identify mid-run utilization drops from a stream of residency samples, and write regression tests in `tests/test_regression.py`.
