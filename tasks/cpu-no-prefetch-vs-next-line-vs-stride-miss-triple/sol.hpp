#pragma once

// 3 deterministic direct-mapped cache models (harness code, defined in
// main.cpp), same line size (64 bytes) and set count (16 sets), but
// different automatic prefetch behavior:
//
//   touch_no_prefetch(addr) -- plain cache, no prefetching.
//   touch_next_line(addr)   -- next-line prefetcher: every MISS on line L
//     also brings line L+1 into the cache as a free, unmeasured
//     side-effect (a fixed hardware policy: always prefetch the next
//     physical line after any miss).
//   touch_stride(addr)      -- stride prefetcher: tracks the delta
//     between this cache's last two accesses; once the SAME delta has
//     been seen twice in a row, it prefetches (address + delta) --
//     predicting the pattern continues -- as a free, unmeasured
//     side-effect of the current access.
//
// All 3 share nothing with each other; reset_prefetch_caches() clears
// all 3 to empty. The miss_count_*() functions read each cache's running
// total misses since the last reset.
void reset_prefetch_caches();
void touch_no_prefetch(long addr);
void touch_next_line(long addr);
void touch_stride(long addr);
long miss_count_no_prefetch();
long miss_count_next_line();
long miss_count_stride();

// generate_and_run: build a strided access trace of n_steps addresses,
// address_k = base + k*stride_bytes for k in [0, n_steps), and feed EACH
// address, IN ORDER, through all 3 functions above -- same address, same
// order, to all three, so the comparison is fair. Then write the 3
// running miss totals into out[0] (no-prefetch), out[1] (next-line),
// out[2] (stride).
void generate_and_run(long base, int stride_bytes, int n_steps, long* out);
