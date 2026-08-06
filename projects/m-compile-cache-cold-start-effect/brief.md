# Ticket: Edge Deployment Latency Spikes and Artifact Mismatches in Fleet Production

## Problem Description
Over the past two release cycles, telemetry from our edge model deployment pipeline indicates severe latency spikes affecting early production traffic immediately following fleet model updates. Field diagnostics report that initial requests on edge devices take upwards of several seconds before normalizing to expected execution times. This behavior degrades user experience and triggers automated fallback mechanisms in downstream services.

Concurrently, several fleet edge devices running low-memory runtimes have reported intermittent crash loops and inconsistent execution results after automatic artifact synchronization. Preliminary inspection suggests that modified model artifacts or corrupted cache blobs are being loaded without validation against the expected release manifest.

## Expected Deliverable
You must build a complete compilation cache, latency estimation, and manifest validation utility in the `edge_cache` package.

1. **Manifest Integrity (`edge_cache/manifest.py`)**: Implement an artifact verification system that calculates cryptographic checksums for model weights, compilation assets, and metadata against a structured manifest dictionary. If any checksum fails or a file is missing, runtime loading must be blocked.
2. **Compile-Cache Cold-Start Simulation (`edge_cache/compiler.py`)**: Implement an edge compilation engine that simulates warm vs. cold cache execution states. The runtime must persist compiled artifacts to disk and load cached compilation objects when present and valid, avoiding costly re-compilation passes.
3. **Population-Weighted p95 Latency Estimator (`edge_cache/metrics.py`)**: Implement an estimator that aggregates cold-start and warm-start execution latencies across diverse device populations to compute population-weighted p95 tail latencies, ensuring telemetry correctly reflects fleet-wide deployment performance.
4. **Safety Regression Suite (`tests/test_regression.py`)**: Write a test suite validating that your integrity verification detects corrupted or truncated model artifacts before cache execution occurs.
