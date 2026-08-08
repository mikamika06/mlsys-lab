# Ticket: Production serving latency spikes and artifact load crashes

## Symptom
During live deployment rollout of dynamic-batch LLM model workers, the latency monitor recorded severe, intermittent p99 latency spikes (up to 300x baseline execution time) when incoming request batches varied in size. The team attempted to switch from `torch.compile` JIT tracking to exported AOT artifacts (`torch.export`), but several edge worker nodes failed to start, crashing with uncaught binary stream parse errors when loading exported packages over network storage.

## Tasks
1. Implement a latency analysis tool (`compilebench/bench.py`) to quantify dynamic recompile overheads in `torch.compile` versus flat dispatch latencies in AOT `torch.export` programs across varying request batch sequences.
2. Implement a robust export artifact parser (`compilebench/artifact.py`) that validates binary headers, table offsets, payload boundaries, and checksum integrity, raising structured error types (`InvalidMagicError`, `TruncatedArtifactError`, `CorruptedArtifactError`) on damaged export packages.
3. Add a regression test suite in `tests/test_regression.py` that asserts artifact corruption detection invariants and catches unvalidated payload deserialization leaks.
