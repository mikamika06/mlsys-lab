#include "sol.hpp"

// TODO: touch every element of the N x N row-major matrix exactly once,
// via touch((row * N + col) * 4), choosing a loop order that keeps
// consecutive touches inside the same cache line as often as possible.
void access_pattern(int N) {
    (void)N;
    // your code here
}
