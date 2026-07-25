#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache (64-byte lines, 8 sets, 4-way -- 2048 bytes
// total). Real hardware cache timing isn't reproducible across machines,
// so this model -- not the CPU's actual cache -- is the sole source of
// every miss count the driver prints. Call touch() once per byte address
// you access.
void touch(long byte_addr);

// The harness's own baseline reads ONE 4-byte field out of an
// Array-of-Structs record array: `record_bytes` bytes per record, N
// records back-to-back, the target field at a fixed `field_offset`
// within each record -- touching aos_base + i*record_bytes + field_offset
// for every i in [0, N). Every touch lands in a DIFFERENT cache line
// (record_bytes is a whole multiple of the 64-byte line size), so every
// record fetches a full line just to read 4 bytes of it.
//
// Touch the SAME N logical values, but from a fresh, CONTIGUOUS
// Struct-of-Arrays array of N 4-byte elements starting at `soa_base`:
// element i lives at soa_base + i*4. Touch every element EXACTLY ONCE,
// in increasing i order -- with nothing else in between, consecutive
// elements share a cache line instead of each needing one of its own.
void soa_field_touch(int N, long soa_base);
