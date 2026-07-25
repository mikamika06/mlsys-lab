#pragma once

// Modeling convention shared with this track's other cache tasks:
// 64-byte lines, and an 8192-byte / 128-line L2 budget.
constexpr int LINE_BYTES = 64;
constexpr long L2_LINE_BUDGET = 128;

// Reuse distance of access i (0-indexed into `addrs`): let j be the
// CLOSEST earlier index (j < i) whose address touches the same 64-byte
// line as addrs[i] (i.e. addrs[j]/64 == addrs[i]/64). The reuse
// distance of access i is the number of DISTINCT 64-byte lines touched
// strictly between j and i -- by some index k with j < k < i -- not
// counting the line itself. If addrs[i] is the FIRST access to its
// line (no such j exists), it contributes nothing (there's no reuse to
// measure yet).
//
// Returns the MAXIMUM reuse distance over the whole trace (0 if no
// line is ever touched twice). Under a fully-associative LRU cache with
// capacity C lines, an access with reuse distance < C is GUARANTEED to
// hit; the max reuse distance is therefore the tightest per-line cache
// size that guarantees every reuse in this trace hits.
long max_reuse_distance(const long* addrs, int n);
