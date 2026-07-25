#pragma once

// Simulate a write-combining (WC) buffer with a fixed number of active
// cache-line slots over a store trace of byte addresses, and write
// {full_flush, partial_flush} counts into out[0], out[1].
//
// Each store address `a` belongs to line = a / line_bytes at byte offset
// a % line_bytes:
//   - If `a`'s line has no active WC entry: if all `slots` entries are
//     already occupied, first evict the OLDEST occupied entry (FIFO by
//     the order lines were first touched) -- flushing it counts a
//     full_flush if every offset in [0, line_bytes) had been recorded
//     for that entry, otherwise a partial_flush. Then open a fresh,
//     empty entry for `a`'s line.
//   - Record offset `a % line_bytes` into that line's entry.
//   - If the entry now has EVERY offset in [0, line_bytes) recorded (the
//     line is full), flush it immediately and count a full_flush.
// After the trace ends, flush every entry still resident, oldest first,
// using the same full/partial rule.
void wc_flush_stats(const long* addrs, int n, int line_bytes, int slots, long* out);
