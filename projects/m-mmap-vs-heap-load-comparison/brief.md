# Ticket #3841: Excessive Memory Consumption and Unexplained Binary Bloat in Edge Deployments

**Severity:** High
**Reporter:** Edge Runtime Performance Team
**Component:** `rw2-edge-export` / Weight Delivery & Selective Builds

### Symptom
When deploying updated vision-language and transformer models to edge devices, field devices report severe memory pressure and unexpected Out-Of-Memory (OOM) crashes during model initialization. Specifically, devices with tight RAM limits (such as 1 GB embedded boards) crash immediately upon weight loading even when only running lightweight sub-graphs or selective builds.

Furthermore, recent candidate model builds show significant binary footprint increases compared to baseline releases, but automated release checks fail to isolate which layers or tensor changes are driving the size growth. Initial profiling suggests our current weight loading paradigm allocates heap memory for the entire tensor file upfront, ignoring page-aligned memory mapping advantages. Additionally, repeated weight tensors (such as shared embeddings and projection blocks across attention heads) appear to consume redundant memory across both disk storage and resident RAM.

### Goal
Implement weight loading footprint analysis (`memload.loader`), size-regression attribution (`memload.attribution`), and weight deduplication metrics (`memload.dedup`) to accurately profile and optimize weight memory overhead across heap vs. mmap loading paradigms. Finally, provide regression tests in `tests/test_regression.py` to prevent memory measurement anomalies and invalid dedup accounting in future releases.
