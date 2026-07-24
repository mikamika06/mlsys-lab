#include "sol.hpp"

void predictor_mispredicts(const int* outcomes, int n, int hist_bits, int* out) {
    // out[0]: always-taken
    int always_mp = 0;
    for (int i = 0; i < n; i++) if (outcomes[i] == 0) always_mp++;

    // out[1]: 1-bit last-outcome
    int last_pred = 0;
    int one_bit_mp = 0;
    for (int i = 0; i < n; i++) {
        if (last_pred != outcomes[i]) one_bit_mp++;
        last_pred = outcomes[i];
    }

    // out[2]: single 2-bit saturating counter
    int cnt = 1;
    int two_bit_mp = 0;
    for (int i = 0; i < n; i++) {
        int pred = (cnt >= 2) ? 1 : 0;
        if (pred != outcomes[i]) two_bit_mp++;
        if (outcomes[i] == 1) cnt = (cnt + 1 > 3) ? 3 : cnt + 1;
        else cnt = (cnt - 1 < 0) ? 0 : cnt - 1;
    }

    // out[3]: gshare
    int table_size = 1 << hist_bits;
    int mask = table_size - 1;
    int* table = new int[table_size];
    for (int i = 0; i < table_size; i++) table[i] = 1;
    int hist = 0;
    int gshare_mp = 0;
    for (int i = 0; i < n; i++) {
        int idx = hist & mask;
        int pred = (table[idx] >= 2) ? 1 : 0;
        if (pred != outcomes[i]) gshare_mp++;
        if (outcomes[i] == 1) table[idx] = (table[idx] + 1 > 3) ? 3 : table[idx] + 1;
        else table[idx] = (table[idx] - 1 < 0) ? 0 : table[idx] - 1;
        hist = ((hist << 1) | outcomes[i]) & mask;
    }
    delete[] table;

    out[0] = always_mp;
    out[1] = one_bit_mp;
    out[2] = two_bit_mp;
    out[3] = gshare_mp;
}
