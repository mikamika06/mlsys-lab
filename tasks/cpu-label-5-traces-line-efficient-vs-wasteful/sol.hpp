#pragma once

// `addrs[0..num_accesses)` is a trace of byte addresses; each access
// reads `elem_bytes` bytes starting there (addresses may repeat -- a
// repeated access is a real re-read of already-cached data, still
// useful work). Grouping addresses into `line_bytes`-byte cache lines:
//
//   bytes_used     = num_accesses * elem_bytes         (every access counts, incl. repeats)
//   bytes_fetched  = (# DISTINCT lines touched across the whole trace) * line_bytes
//   efficiency     = bytes_used / bytes_fetched
//
// Return 1 ("line-efficient") if efficiency >= 0.5, else 0 ("wasteful").
int classify_trace(const long* addrs, int num_accesses, int elem_bytes, int line_bytes);
