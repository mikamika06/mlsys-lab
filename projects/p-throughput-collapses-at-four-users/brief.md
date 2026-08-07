# Throughput collapses at four concurrent users

## Incident Report
Our local model runner (`runner`) exhibits unexpected throughput behavior as concurrent user load increases. During single-user benchmarks, the engine delivers approximately 38 tok/s per stream. With two concurrent users, aggregate throughput reaches around 68 tok/s (about 34 tok/s per stream), scaling almost linearly as expected.

However, when four concurrent users send requests simultaneously, performance degrades severely. The aggregate throughput collapses to under 12 tok/s, request queues begin to back up rapidly, and tail latency (p95) spikes by several orders of magnitude.

Engineers expected throughput to scale linearly with concurrent streams up to the hardware compute limit, but the system hits a sharp bottleneck ("knee") much earlier than anticipated.

## Goal
Investigate the local runner's performance under concurrent load. You need to build a load benchmarking harness with proper warmup, construct the throughput vs. concurrency curve to locate the knee, decompose execution timing (queue wait, prefill, and decode), identify the root bottleneck limiting slot concurrency, optimize the runner configuration to shift the knee out by at least +1 user without breaching p95 latency targets, and build an analytical queueing model to predict p95 latency at 8 users within ±15% accuracy.
