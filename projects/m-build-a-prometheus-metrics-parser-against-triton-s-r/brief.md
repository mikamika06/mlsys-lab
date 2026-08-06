# Prometheus Metrics Parser Against Triton's Real Metric Names

Production monitoring of Triton Inference Server relies on scraping its HTTP/Prometheus endpoint (typically port 8002). The metrics endpoint exposes raw text lines in standard Prometheus exposition format containing real Triton counter, gauge, and summary/histogram metrics such as `nv_inference_request_success`, `nv_inference_compute_infer_time_us`, and `nv_gpu_utilization`.

We recently noticed our telemetry dashboard losing track of per-model request throughput and GPU usage spikes. SRE reports that key metrics are either missing from aggregated metrics or producing incorrect values when labels like `model`, `version`, or `gpu_uuid` are present. Furthermore, automated load alerts failed to trigger because latency summary counters were miscalculated during scrape parsing.

Your task is to implement a robust Prometheus exposition format parser and metrics aggregator tuned to Triton's actual metric names and labels.

1. Implement `triton_metrics/parser.py` to parse raw Prometheus metrics text into structured `MetricSample` records. Handle comments, HELP/TYPE annotations, sample lines with and without label sets, floating point / infinite values, and microsecond-to-millisecond/second conversions where requested.
2. Implement `triton_metrics/aggregator.py` to derive operational summaries across parsed scrapes: extract per-model request counts, compute average compute latency per model, calculate per-GPU utilization statistics, and identify anomalous latency spikes across model versions.
3. Write a suite of regression tests in `tests/test_regression.py` that validates metric aggregation invariants and verifies that malicious or corrupt Prometheus metric lines do not lead to silent data contamination or erroneous request counts.
