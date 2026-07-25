#pragma once

// A stride-table prefetcher tracks, PER STREAM (indexed by stream_id,
// 0..num_streams), the last address it saw for that stream and the
// delta from the access before that. Given n accesses, where access i
// is (stream_id[i], addr[i]):
//
//   - The FIRST access of a given stream: just remember its address (no
//     delta yet, nothing to compare, no prefetch).
//   - The SECOND access of a given stream: compute
//     delta = addr[i] - last_addr[stream]; remember it as that stream's
//     "recorded delta" (no prefetch yet -- one delta alone does not
//     confirm a pattern).
//   - Every access after that: compute delta the same way; if it EQUALS
//     the stream's currently recorded delta, the pattern is confirmed
//     -- issue a prefetch for (addr[i] + delta) and count it. Either
//     way, update the stream's recorded delta to this access's delta.
//   - After every access (of any kind above), update the stream's last
//     address to addr[i].
//
// Different streams' state must never mix, even though the accesses of
// different streams are interleaved in the trace. Return the TOTAL
// number of prefetches issued across all streams.
long stride_prefetch_count(const int* stream_id, const long* addr, int n, int num_streams);
