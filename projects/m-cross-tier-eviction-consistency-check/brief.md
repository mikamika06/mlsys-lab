# Cross-Tier Eviction Consistency Check

We are observing intermittent cache corruption and stale state issues in our multi-tier KV cache serving engine. The serving system maintains a fast primary Tier-0 cache (GPU/HBM) alongside a secondary Tier-1 cache (DRAM or Host SSD). During high load, blocks evicted from the primary tier are expected to be asynchronously transferred or synchronized with the secondary tier before their slots are reused.

However, recent system traces show that under fast-path eviction triggers, the primary tier releases block references without ensuring the corresponding secondary tier entries are either properly updated, marked for synchronization, or explicitly invalidated. This leads to dangling references, stale cache reads across tiers, or inconsistent block states when attempting to reload cache blocks back into Tier-0.

Your task is to implement a robust, cross-tier eviction consistency checker and manager. The manager must track block state transitions across primary and secondary tiers, validate cross-tier consistency rules during eviction events, and generate consistent eviction plans. Finally, you will write a suite of regression tests to safeguard the cross-tier eviction logic against common synchronization bugs.
