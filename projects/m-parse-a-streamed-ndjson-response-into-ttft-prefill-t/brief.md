# Streamed NDJSON Timing Analysis & Bottleneck Diagnostics

We have received multiple bug reports regarding inaccurate inference latency metrics and unreliable benchmarking in our runner framework. Users report that time-to-first-token (TTFT), prefill throughput (tokens/sec), and decode throughput (tokens/sec) calculations drift significantly when streaming model responses, leading to faulty load-balancer routing and wrong scaling decisions.

Specifically, the current response parser fails to handle chunked Server-Sent Events / NDJSON streaming frames, losing precision when tracking token production events. Additionally, our benchmarking runner does not isolate cold-start warmup effects when aggregating aggregate decoding speeds over multi-step evaluations. Finally, our automated capacity planner cannot reliably classify whether a given inference trace is bound by input token processing (prefill-dominated) or generation steps (decode-dominated).

Your task is to build a timing parse and telemetry evaluation package that reads streamed NDJSON response logs, computes exact latency counters (TTFT, prefill tok/s, decode tok/s), builds a benchmark framework filtering warmup spikes to report median decode throughput, and diagnoses whether incoming operational traces are prefill- or decode-dominated based on timing counters.
