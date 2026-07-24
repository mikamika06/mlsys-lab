#include "sol.hpp"

void process_shifts(const int* values, const int* shift_amounts, long* results, int n) {
    for (int i = 0; i < n; i++) {
        unsigned int uv = (unsigned int)values[i];
        unsigned int amt = (unsigned int)shift_amounts[i] % 32u;
        unsigned int ur = uv << amt;
        int sr = (int)ur;
        results[i] = (long)sr;
    }
}
