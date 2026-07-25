#include "sol.hpp"

// BUG: naive i-j-k order, no blocking at all -- for every single row i
// of Q, the entire j range (all S rows of K) is re-streamed from
// scratch, so K (which is bigger than the cache on its own) never stays
// resident across rows of Q.
void qkt_access(int S, int d, int B, int elem_bytes) {
    (void)B;
    long baseK = (long)S * d * elem_bytes;
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            for (int k = 0; k < d; k++) {
                touch((long)(i * d + k) * elem_bytes);
                touch(baseK + (long)(j * d + k) * elem_bytes);
            }
        }
    }
}
