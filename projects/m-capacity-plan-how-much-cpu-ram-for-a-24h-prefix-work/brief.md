# Symptom Brief: Out-of-Memory and Unexpected Storage I/O Spikes in Long-Context Serving

During production deployment of our multi-tenant KV cache tier, serving nodes experience severe performance degradation and random OOM kills under 24-hour continuous traffic workloads. Monitoring indicates that our system severely miscalculates the CPU RAM capacity required to retain warm prompt prefixes across a rolling 24-hour window, leading to aggressive cache evictions.

Furthermore, when warm prefixes are offloaded to secondary disk-backed NVMe storage, disk read throughput spikes dramatically higher than the logical byte count of requested KV blocks. Systems engineers suspect that the disk fetch layer suffers from massive read amplification due to unaligned block lookups and fixed storage chunking, but we currently lack formal measurement tools or accurate sizing models.

You need to build a capacity planning module that accurately calculates total CPU RAM requirements given model configuration, token reuse metrics, and target prefix retention windows. Additionally, implement an accurate disk-backed cache tracer to measure physical bytes read versus logical KV bytes requested, exposing the exact read amplification factor across sliding page aligned boundaries.
