Title: KV cache OOMs rapidly on beam search.

Description: We are experiencing catastrophic OutOfMemory (OOM) errors on the production inference server. This happens specifically when users enable beam search or parallel sampling. The server is provisioned with enough memory to hold tens of thousands of tokens in the KV cache, yet it crashes even with small batch sizes if branching is involved.

For instance, feeding a 2000-token prompt and generating 50 tokens with 8 beams instantly exhausts the block pool. The telemetry indicates that the moment the branching occurs, the number of free blocks drops precipitously, as if the entire prompt's history is being consumed multiple times over. Furthermore, after extensive parallel generation and cleanup, the free block count does not always return to the expected baseline, suggesting potential leaks or severe fragmentation over time.

Our current allocator simply assigns blocks sequentially. Please redesign the KV block allocation subsystem so that it can handle branched generation gracefully without blowing up the memory budget. The system must survive long-running traces of 10k+ operations without leaks, and you should ensure proper tracking of active references so we don't accidentally free data that is still in use by another branch.
