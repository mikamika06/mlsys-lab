#include "sol.hpp"

void temporal_memset(long base, long nbytes, int line_bytes) {
    int n = (int)(nbytes / line_bytes);
    for (int k = 0; k < n; k++) {
        touch(line_addr(base, line_bytes, k));
    }
}
