#include "sol.hpp"

// TODO: sort the field sizes descending (largest alignment first -- this
// is provably optimal when each field's alignment equals its size), then
// lay them out: each field at the next offset that's a multiple of its own
// size, total rounded up to a multiple of the largest size. See sol.hpp.
int minimal_sizeof(const int* sizes, int n) {
    (void)sizes;
    (void)n;
    // your code here
    return 0;
}
