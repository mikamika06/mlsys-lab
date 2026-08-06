# Ticket: Memory diagnostics inconsistencies during heavy model runs

We are currently experiencing unexplained out-of-memory (OOM) exceptions and memory usage discrepancies during large-scale transformer fine-tuning runs on multi-GPU setups. Specifically, the reported peak allocated memory from standard monitoring tools does not match the actual memory pressure observed right before a CUDA OOM crash occurs, leading to sudden failures during long training iterations.

Additionally, profiling logs exported via PyTorch memory timeline features show unexpected memory footprints that do not align with our expected tensor allocations. Engineers note that fragmentation within the caching allocator might be causing blocks to remain reserved or unsplit improperly, even when allocations appear inactive or small. However, without a precise programmatic way to parse these timeline exports, identify the true largest live allocation at the moment of failure, and accurately simulate block splitting and merging behavior, we cannot reliably audit memory efficiency or predict fragmentation overheads.

We need a robust module to programmatically inspect memory timeline exports, extract true peak allocated bytes, locate the precise largest active allocation right before a recorded OOM event, and simulate caching allocator fragmentation metrics to prevent these unexpected crashes.
