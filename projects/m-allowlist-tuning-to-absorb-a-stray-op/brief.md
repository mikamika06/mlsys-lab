Ticket: ENG-4922
Component: Edge Export Pipeline

We are tracking a severe memory explosion issue when exporting graphs to the new NPU delegate. The core symptom is that compilation of standard 12-op sequence blocks fails because the memory footprint of the delegate blobs exceeds the device budget. 

Looking at the traces, our graph partitioner is chopping sequences of 12 ops into multiple disconnected delegate blobs. This happens because a single, unsupported "stray" op (like a `Cast` or `Reshape`) gets sandwiched between supported `Conv2D` sequences. Every new delegate blob introduces a massive byte overhead for the blob descriptor and NPU initialization state.

A firmware engineer suggested: "If we just allowlist that one stray op, we can absorb it into the blob. The NPU will fall back to a slower kernel for that op, adding some baseline bytes to the blob, but we save the massive overhead of instantiating multiple blobs."

We need an automated tuning pass. 
Task 1: Implement a predictor that assigns delegate blob IDs to a 12-op sequence based on a given allowlist.
Task 2: Build the byte accounting logic. Given a base allowlist and some candidate ops, evaluate if adding one candidate to the allowlist reduces the total byte cost (sum of delegated op sizes + per-blob overhead). Return the optimal candidate and the resulting byte cost.
