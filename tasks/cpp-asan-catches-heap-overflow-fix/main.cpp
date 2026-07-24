#include <cstdio>
#include "sol.hpp"

int main() {
    const int NF = 3;
    int fixtures[NF] = {4, 1, 7};

    for (int f = 0; f < NF; f++) {
        int n = fixtures[f];
        // Real heap allocation: n logical chunks + 1 hidden guard chunk
        // right after them, at index n.
        DataChunk* chunks = new DataChunk[n + 1];
        for (int i = 0; i <= n; i++) {
            chunks[i].header = -999;
            for (int j = 0; j < 4; j++) chunks[i].values[j] = -999.0;
        }

        populate_chunks(chunks, n);

        for (int i = 0; i < n; i++) {
            printf("%d:", chunks[i].header);
            for (int j = 0; j < 4; j++) printf("%.3f,", chunks[i].values[j]);
        }
        // Guard chunk must remain untouched (-999 sentinel) — this is what
        // catches an off-by-one overflow.
        printf(" guard=%d:", chunks[n].header);
        for (int j = 0; j < 4; j++) printf("%.3f,", chunks[n].values[j]);
        printf("\n");

        delete[] chunks;
    }
    return 0;
}
