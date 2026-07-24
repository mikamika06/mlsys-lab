#pragma once

// False sharing: two threads writing to DIFFERENT variables that happen
// to land in the same cache line still bounce that line between cores as
// if it were one shared variable. True sharing (multiple threads writing
// the exact same address) is a real data dependency, not a false-sharing
// bug, and must NOT be reported.
//
// Given n writes, where write i is `addrs[i]` performed by thread
// `thread_id[i]`, group writes by cache line (`addrs[i] / line_bytes`).
// A line is FALSELY SHARED iff:
//   - writes to it come from >= 2 distinct thread ids, AND
//   - those writes touch >= 2 distinct addresses (not all the same byte
//     address).
//
// Write the falsely-shared line ids, sorted ascending with no duplicates,
// into `out` (the caller guarantees `out` has room for at least n
// entries) and return how many were written.
int find_falsely_shared_lines(const long* addrs, const int* thread_id, int n, int line_bytes, long* out);
