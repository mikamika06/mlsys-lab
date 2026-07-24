#pragma once

// Cache-hierarchy access hook, DEFINED in main.cpp: a small deterministic
// 3-level INCLUSIVE cache model --
//   L1: direct-mapped,  8 sets, 64-byte lines  (512 bytes total)
//   L2: 4-way set-assoc, 16 sets, 64-byte lines (4096 bytes total)
//   L3: 8-way set-assoc, 32 sets, 64-byte lines (16384 bytes total)
// -- with inclusion enforced: evicting a line from L2 invalidates it from
// L1 too (if present); evicting a line from L3 invalidates it from L2 and
// L1 too. Real hardware cache timing is not reproducible across machines,
// so this model (not the CPU's real cache) is the ONLY source of the miss
// counts the driver prints. Call touch() once per byte address you access;
// the driver reads the model's per-level miss counters afterward.
void touch(long byte_addr);

// Touch every element of an N x N matrix of 4-byte elements, stored
// ROW-MAJOR (element (row, col) lives at byte address
// (row * N + col) * 4), by calling touch() on every element's address
// EXACTLY ONCE. The loop order you choose is the whole exercise -- the
// cache model notices.
void access_pattern(int N);
