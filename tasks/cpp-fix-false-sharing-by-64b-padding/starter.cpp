#include "sol.hpp"

// BROKEN: no padding at all. sizeof(ThreadData) == 8, so all 4 threads'
// counters (addresses 0, 8, 16, 24) land in the same 64-byte cache line —
// every thread after the first is a false-sharing write.
struct ThreadData {
    long counter;
};

int thread_data_sizeof() {
    return static_cast<int>(sizeof(ThreadData));
}
