#include "sol.hpp"

struct ThreadData {
    long counter;
    long pad[7];   // 7 * 8 = 56 bytes, so sizeof(ThreadData) == 64
};

int thread_data_sizeof() {
    return static_cast<int>(sizeof(ThreadData));
}
