# Ticket: Unexplained All-Gather Overheads During State Dict Checkpointing

## Symptom
During distributed training checkpoint saves, our FSDP2 training runs occasionally spike in inter-node communication traffic and stall for tens of seconds when dumping sharded state dicts. The team expected sharded state dict operations to be completely communication-free across ranks since each GPU already holds its local DTensor parameter slice.

Profiling shows two related anomalies. First, calling state dict serialization triggers non-zero collective calls on certain model subtrees. Second, whenever engineers alter module wrapping sequences or manually register sub-modules out of order, the memory allocation during unshard/shard transitions jumps unexpectedly, and state dict extraction reverts to triggering implicit rank synchronization.

## Work Needed
We need a diagnostic and verification toolset in `fsdp_verify/` to:
1. Audit sharded state dict metadata against worker DTensor placements to verify zero rank-to-rank communication is required, identifying non-local slice dependencies.
2. Quantify bottom-up wrap order violations by measuring depth mismatches and illegal execution edge dependencies across sub-modules.
3. Automatically reconstruct the canonical bottom-up `fully_shard()` call sequence for any arbitrary nested model graph so wrapping order guarantees are preserved.
4. Provide regression tests in `tests/test_regression.py` that actively detect misordered module wrapping schedules.
