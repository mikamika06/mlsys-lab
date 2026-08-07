# INCIDENT: OOM with Activation Checkpointing

**Reporter:** @syseng-lead
**Component:** Training / Checkpointing

We are hitting out-of-memory (OOM) errors when training a 100-layer transformer. We enabled gradient checkpointing, expecting peak memory to drop significantly. However, memory spikes way higher than our spreadsheet predicts. I suspect our segment sizing is sub-optimal, or our mathematical model of peak memory completely forgets that recomputing a segment requires instantiating its intermediate activations simultaneously with the saved checkpoints!

Could you build a rigid simulator to calculate the exact peak memory and step time for a given segment count?

1. Implement `baseline(n_layers, layer_mem, fwd_time, bwd_time)` computing memory and time without checkpointing (all intermediate activations saved).
2. Implement `simulate_checkpointing(n_layers, segments, layer_mem, fwd_time, bwd_time)`. Split the layers into `segments` groups (as evenly as possible, putting larger segments first if they do not divide evenly). Simulate the memory lifecycle: allocating checkpoints during the global forward, and allocating intermediates during recomputation in the backward pass.
3. Implement `optimal_segments(n_layers)` that uses your simulator to find the segment count `S` yielding the lowest peak memory. If tied, prefer the smaller `S`.
4. Provide a regression test that proves recomputation actually costs memory. Our current broken spreadsheet effectively ignores intermediate memory during recompute.
