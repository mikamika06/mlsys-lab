# Incident Report: Low Acceleration Ratio on TensorRT Execution Provider

## Symptom
Production inference deployment for our vision-language backbone was configured to run with the TensorRT Execution Provider (TRT EP) enabled on NVIDIA GPU instances. However, operational telemetry reports an unacceptable performance speedup of only 8% over the baseline CPU execution provider, far below the projected 3.5x speedup target.

Additionally, service start times after pod restarts show severe tail latencies, taking over 45 seconds before the first request can be served.

## Profiling Observations
Initial trace inspections reveal that execution switches back and forth between host RAM and GPU VRAM hundreds of times per single inference pass. The graph seems to be broken into dozens of micro-subgraphs instead of running as a unified GPU kernel execution pipeline. Furthermore, every process restart rebuilds all GPU engine artifacts from scratch, causing massive startup delays.

## Task
Audit the graph partitioning pipeline to diagnose why the model is failing to run on TensorRT EP. You need to inspect the partitioned subgraphs, identify the exact unsupported ops causing partitioning breaks, implement a graph rewriting pass to substitute or reorder unsupported nodes, and integrate an engine cache to eliminate redundant cold-start build overhead.
