#pragma once

// Deterministic set-associative LRU byte-cache model, defined in main.cpp:
// 64-byte lines, 32 sets, 4-way (8192 bytes total). touch_byte(addr)
// simulates reading/writing the byte at address `addr` through this cache
// and counts a MISS whenever that cache line wasn't already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// A decode-time KV cache holds T tokens, H attention heads, a key/value
// selector k in {0, 1}, and D floats of head dimension, each element E
// bytes. The tested shapes satisfy 2*D*E == 64, so one head's complete K+V
// record for ONE token fits exactly in one cache line.
//
// Two access patterns compete for the same physical layout:
//   - whole-token write: appending the newest token t = T-1 touches every
//     (h, k, d) for that one t, in order h, then k, then d.
//   - per-head decode read: streaming one head's K/V across every existing
//     token touches every (t, k, d) for that one h, in order t, then k,
//     then d, for h = 0..H-1.
//
// Implement thkd_addr as the TOKEN-major layout: token, then head, then
// key/value, then dimension (layout id "THKD"). This keeps a whole token's
// multi-head record contiguous (cheap write) while still keeping each
// individual head's one-line K+V record intact (cheap per-head read, since
// that one line is never shared with another head).
//
//   index = ((( t*H + h )*2 + k )*D + d)
//   addr  = base + index * E
long thkd_addr(long base, int T, int H, int D, int E, int t, int h, int k, int d);
