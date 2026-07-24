#pragma once

// Simulate a fully-associative LRU cache holding at most `capacity`
// distinct line ids, processing the access trace `ids[0..n)` (ids are
// abstract cache-line identifiers; a repeated id is a re-access of the
// same line, not a new one). On each access:
//   - if the id is currently resident, it's a HIT, and it becomes the
//     most-recently-used resident id;
//   - otherwise it's a MISS: the id is inserted as most-recently-used,
//     evicting the current least-recently-used resident id first if the
//     cache already holds `capacity` ids.
// Return the total number of MISSES over the whole trace.
long lru_miss_count(const int* ids, int n, int capacity);
