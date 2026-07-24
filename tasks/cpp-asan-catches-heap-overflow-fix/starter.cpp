#include "sol.hpp"

// BUG: off-by-one loop bound. `i <= num_chunks` also writes chunks[num_chunks],
// one past the num_chunks logical elements — a heap buffer overflow that
// corrupts the guard chunk the caller checks. Fix the bound.
void populate_chunks(DataChunk* chunks, int num_chunks) {
    for (int i = 0; i <= num_chunks; i++) {
        chunks[i].header = i;
        for (int j = 0; j < 4; j++)
            chunks[i].values[j] = i * 1.5 + j;
    }
}
