#pragma once

// Pinned line size (defined in main.cpp). h_bytes and a_bytes below are
// always exact multiples of this.
extern const int LINE_BYTES;

// Deterministic direct-mapped cache model (harness code, defined in
// main.cpp): LINE_BYTES-byte lines, 256 sets, 1 way per set.
// reset_cache() clears it. touch_byte(addr) is an ORDINARY ("temporal")
// access through this cache -- it returns true iff the access MISSED
// (the line wasn't already resident), and either way the line becomes
// resident afterward, evicting whatever else shared its set.
void reset_cache();
bool touch_byte(long addr);

// nontemporal_store(addr): a real streaming/non-temporal store. It
// writes straight to memory through a write-combining buffer and NEVER
// consults or updates the cache model above: no residency, no
// eviction, and -- since the whole line is being overwritten anyway --
// no read-for-ownership.
void nontemporal_store(long addr);

// Models writing out a large "activation" buffer A of `a_bytes` bytes
// (produced once, not read back soon) while a separate "hot" tensor H
// of `h_bytes` bytes (already resident, and read again immediately
// after A is written) is still in active use elsewhere. Byte addresses:
// H occupies [0, h_bytes), A occupies [h_bytes, h_bytes + a_bytes).
// MUST begin by calling reset_cache(). Then, one LINE_BYTES-sized line
// at a time, in increasing address order:
//
//   1. Warm H: touch_byte() every line of H once. H starts cold, so
//      every line misses; charge LINE_BYTES DRAM bytes per miss (a
//      plain read-fill).
//   2. Write A, once:
//        - use_nontemporal == true:  nontemporal_store() every line of
//          A. Charge exactly LINE_BYTES per line (a pure write -- no
//          read-for-ownership, since streaming stores skip the cache).
//        - use_nontemporal == false: touch_byte() every line of A.
//          These addresses are cold, so every line misses, and because
//          an ordinary store is write-allocate, charge 2*LINE_BYTES per
//          miss (the read-for-ownership fetch, PLUS the line's
//          eventual writeback) -- twice the DRAM traffic of writing the
//          same line non-temporally.
//   3. Re-touch H: touch_byte() every line of H again. If step 2
//      evicted that line (only possible when use_nontemporal was
//      false), this misses; charge LINE_BYTES per miss (a pollution
//      refetch). A hit charges 0.
//
// Return the total DRAM bytes charged across all three steps.
long modeled_dram_traffic(long h_bytes, long a_bytes, bool use_nontemporal);
