#pragma once
#include <string>
#include <utility>
#include <vector>

struct SyncPattern {
    std::vector<std::string> reads;
    std::vector<std::string> writes;
    std::vector<std::string> atomics;
    std::vector<std::pair<std::string, std::string>> locks;      // (a, b); a == b flags a self-lock
    std::vector<std::pair<std::string, std::string>> lock_edges; // (held, requested) lock-order edges
};

// Classify each of n patterns as "ok", "data-race", or "deadlock", in this
// exact priority order:
//   1. If any entry of `locks` has equal elements (a self-lock) ->
//      "deadlock".
//   2. Build a directed graph from `lock_edges` (held -> requested). If it
//      contains a cycle reachable from any node that has an outgoing edge
//      -> "deadlock".
//   3. Let shared = (reads union writes) minus atomics — every non-atomic
//      name touched at all. If any name appears in BOTH `writes` and
//      `reads`, and it is in `shared` (i.e. not atomic) -> "data-race".
//   4. Otherwise -> "ok".
// Write out[i] for i in [0, n).
void classify_sync_patterns(const SyncPattern* patterns, int n, std::string* out);
