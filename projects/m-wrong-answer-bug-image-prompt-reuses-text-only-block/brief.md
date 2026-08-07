An incident report from our prefix-caching serving tier shows intermittent, hard-to-reproduce output corruption during multimodal inference requests. Users submit requests that combine an image prompt and subsequent text queries. Under high load with prefix caching enabled, a specific sequence of prompts triggers an error where the KV cache block allocator maps a text-only block hash onto an image prompt's block, causing the model to attend to incorrect visual tokens or completely corrupted latent representations.

Investigation reveals that when computing hash keys for cache blocks, the serialization path for multimodal prompts occasionally strips out the image modality metadata or handles padding bytes inconsistently if a text-only block matches a truncated prefix hash of an image block. Furthermore, our per-tenant cache quota simulator fails to account for these collisions, leading to inaccurate memory accounting, quota overages, and unexpected eviction of active tenant blocks.

Your task is to investigate and fix this bug across three milestones:
1. Fix the cache block hashing and serialization logic so that image prompts and text-only blocks never produce colliding block hashes, even under truncated hash spaces.
2. Implement and verify a robust collision search mechanism under a truncated hash scheme to detect and resolve prefix collisions safely.
3. Build a per-tenant cache quota simulator that correctly accounts for exact block ownership and prevents multi-tenant isolation breaches.
