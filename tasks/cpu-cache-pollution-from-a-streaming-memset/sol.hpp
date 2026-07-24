#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache (64-byte lines, 16 sets, 4-way -- 4096 bytes
// total capacity). Real hardware cache timing isn't reproducible across
// machines, so this model -- not the CPU's actual cache -- is the sole
// source of every eviction count the driver prints. Call touch() once per
// byte address a REGULAR (temporal) store or load touches. A
// non-temporal / streaming store writes through a write-combining buffer
// straight to memory and never allocates a line in the cache, so it must
// NOT call touch() at all.
void touch(long byte_addr);

// Byte address of the k-th line of `line_bytes` bytes in a region
// starting at byte address `base`.
inline long line_addr(long base, int line_bytes, int k) {
    return base + (long)k * line_bytes;
}

// Simulate a bulk memset of `nbytes` bytes starting at byte address
// `base`, using ORDINARY (temporal) stores. A temporal store allocates
// and fills a cache line exactly like any normal write does, so writing
// this region line-by-line must call
//   touch(line_addr(base, line_bytes, k))
// for every k in [0, nbytes / line_bytes), in increasing order, exactly
// once each -- one touch per line, no more, no fewer.
void temporal_memset(long base, long nbytes, int line_bytes);
