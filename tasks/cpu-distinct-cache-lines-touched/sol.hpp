#pragma once

// Two byte addresses fall in the same cache line iff addr / line_bytes
// (integer division) is equal. Given a trace of n byte addresses, return
// the number of DISTINCT cache lines touched (repeat addresses, or
// different addresses that fall in the same line, count once).
long count_distinct_lines(const long* addrs, int n, int line_bytes);
