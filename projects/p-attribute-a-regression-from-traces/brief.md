# Incident Report: Nightly Benchmark Performance Regression

## Symptom Overview
Our automated nightly performance benchmark suite has reported an unexpected and severe regression: a **+40% increase in total step execution time** on our primary large language model training workload. A comprehensive audit of version control logs confirms that **no code commits, model architecture updates, or hyperparameter modifications** were introduced between yesterday's clean baseline run and today's regressed execution. Furthermore, the underlying compute nodes, driver versions, software library releases, and hardware cluster topology remain entirely unchanged across both timestamps.

## Available Diagnostic Artifacts
To assist in root-cause analysis, the telemetry system has captured detailed event logs and timeline traces from yesterday's stable run (`trace_yesterday.json`) and today's regressed execution (`trace_today.json`), alongside a targeted diagnostic probe trace (`trace_probe.json`). These trace files encapsulate kernel execution durations, host-device launch timestamps, invocation counts, asynchronous queue states, and synchronization barriers.

## Investigation Objectives
Your task is to implement a robust, automated trace analysis module within the profiling framework to systematically isolate and attribute this performance degradation. Specifically, you must accomplish the following operational milestones:

1. **Trace Normalization:** Parse and normalize heterogeneous raw execution traces from multiple runs into clean, structurally comparable tabular summaries mapping kernel names to aggregate counts, total durations, and self-times.
2. **Delta Quantification:** Compare the normalized tables across runs to compute precise self-time deltas and identify the specific kernels or operations responsible for the largest performance regressions.
3. **Operational Profiling:** Characterize the regressed workloads to distinguish whether the bottleneck stems from compute-bound execution limits or launch-bound latency bottlenecks.
4. **Synchronization Detection:** Inspect the trace event streams and timing intervals to uncover hidden host-device synchronization barriers, stream stalls, or unexpected idle bubbles injected into the execution pipeline.
5. **Root Cause Confirmation:** Verify the diagnosed hypothesis against a third confirmation trace (`trace_probe.json`) to validate that the identified bottleneck completely accounts for the observed behavior.
6. **Automated CI Safeguard:** Formulate and ship a rigorous regression test suite for continuous integration that successfully defends against breaking modifications to trace comparator logic and prevents silent regressions.

Follow the milestone criteria carefully to ensure correct implementation and complete diagnostic coverage.
