# Incident Report: Latency Spikes in Paged 8-Bit AdamW Fine-Tuning Runs

## Symptom
During fine-tuning runs utilizing 8-bit paged AdamW (`PagedAdamW8bit`), intermittent performance degrade is observed where single optimizer steps stall execution. While baseline steps complete within expected time bounds, periodic steps suffer from high CPU-to-GPU page fault overheads when state memory overflows host-device boundaries. Profiling logs captured via standard JSON event traces contain recorded host-side and device-side timelines, but manual inspection across multi-step runs is impractical due to event density.

## System Context
The engine logs recorded execution segments as JSON standard trace event arrays. Each entry stores timing details alongside execution metadata including step counters, event names, categories, and memory transfer page fault metadata under `args`. Paged optimizer thrashing exhibits clear non-linear spikes in relative spillover latency where CPU allocation latency dominates active kernel execution.

## Objective
Develop a deterministic parser package `gputrace` that ingests JSON execution traces, isolates `PagedAdamW8bit` execution events, computes CPU-spillover latency metrics, and extracts the index corresponding to the most severe latency spike.

1. Implement `parse_trace_events` in `gputrace/parser.py` to filter trace entries and extract paged optimizer execution records.
2. Implement `find_spillover_spike` in `gputrace/metrics.py` using `np.argmin` over negative efficiency metrics to isolate the exact spillover spike index.
3. Construct regression tests in `tests/test_regression.py` validating trace extraction accuracy and ensuring sensitivity against fault-handling regressions.
