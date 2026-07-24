#include <cstdio>
#include <map>
#include "sol.hpp"

// FIXED driver. Models 4 threads, each "owning" and writing data[thread_id]
// .counter. No real threads and no timing (both non-deterministic) — just a
// deterministic cache-coherence proxy: for each thread in order, compute
// which 64-byte line its counter falls in; if that line is already owned by
// a DIFFERENT thread, that is a false-sharing write.

int main() {
    int stride = thread_data_sizeof();

    std::map<long, int> owner;   // 64B line index -> owning thread id
    int shared_writes = 0;

    for (int tid = 0; tid < 4; ++tid) {
        long addr = static_cast<long>(tid) * stride;
        long line = addr / 64;
        auto it = owner.find(line);
        if (it != owner.end() && it->second != tid) {
            ++shared_writes;
        }
        owner[line] = tid;
    }

    printf("stride=%d\n", stride);
    printf("shared_writes=%d\n", shared_writes);
    return 0;
}
