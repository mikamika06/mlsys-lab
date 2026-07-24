#include "sol.hpp"

// Fixed: loop bound is strictly less than num_chunks, so the guard chunk at
// chunks[num_chunks] is never touched.
void populate_chunks(DataChunk* chunks, int num_chunks) {
    for (int i = 0; i < num_chunks; i++) {
        chunks[i].header = i;
        for (int j = 0; j < 4; j++)
            chunks[i].values[j] = i * 1.5 + j;
    }
}
