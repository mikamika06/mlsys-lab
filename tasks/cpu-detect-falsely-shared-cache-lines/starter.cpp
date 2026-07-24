#include "sol.hpp"

// TODO: group writes by cache line (addrs[i] / line_bytes); a line is
// falsely shared iff its writes come from >= 2 distinct thread ids AND
// touch >= 2 distinct addresses. Write the sorted, deduplicated line ids
// into out[] and return how many were written. See sol.hpp.
int find_falsely_shared_lines(const long* addrs, const int* thread_id, int n, int line_bytes, long* out) {
    (void)addrs; (void)thread_id; (void)n; (void)line_bytes; (void)out;
    // your code here
    return 0;
}
