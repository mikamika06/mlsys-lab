#include <cstdio>
#include "sol.hpp"

static void run_case(int C, const long long* keys, int nkeys) {
    Slot* table = new Slot[C];
    for (int i = 0; i < C; i++) table[i].occupied = false;

    printf("C=%d bytes=%zu slots=", C, (size_t)C * sizeof(Slot));
    for (int i = 0; i < nkeys; i++) {
        int slot = insert_probe(table, C, keys[i]);
        printf(" %d", slot);
    }
    printf("\n");
    delete[] table;
}

int main() {
    long long k1[5] = {10, 20, 30, 40, 50};
    run_case(8, k1, 5);

    long long k2[6] = {3, 8, 13, 18, 23, 1};
    run_case(7, k2, 6);

    long long k3[7] = {100, 200, 300, 400, 500, 600, 700};
    run_case(13, k3, 7);

    return 0;
}
