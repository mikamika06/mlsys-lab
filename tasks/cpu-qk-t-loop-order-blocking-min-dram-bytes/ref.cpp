#include "sol.hpp"

// Tiled: outer loops walk B x B tiles of (i, j); for each tile, every
// (i, j) pair in it runs its full k reduction before moving to the next
// tile. A B-row block of K gets reused across B rows of Q before the
// next block of K is ever touched, instead of the whole of K being
// re-streamed from scratch for every single row of Q.
void qkt_access(int S, int d, int B, int elem_bytes) {
    long baseK = (long)S * d * elem_bytes;
    for (int ii = 0; ii < S; ii += B) {
        for (int jj = 0; jj < S; jj += B) {
            for (int i = ii; i < ii + B; i++) {
                for (int j = jj; j < jj + B; j++) {
                    for (int k = 0; k < d; k++) {
                        touch((long)(i * d + k) * elem_bytes);
                        touch(baseK + (long)(j * d + k) * elem_bytes);
                    }
                }
            }
        }
    }
}
