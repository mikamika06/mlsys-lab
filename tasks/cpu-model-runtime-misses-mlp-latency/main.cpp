#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache level, used only to produce a
// realistic, non-hardcoded miss count for each scenario below -- it is
// plumbing, not the object under test. Real hardware cache timing isn't
// reproducible across machines, so this model is the sole source of
// every miss count.
struct Level {
    int nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;

    Level(int ns, int w) : nsets(ns), ways(w), sets(ns) {}

    void access(long line) {
        auto& s = sets[(int)(line % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
    }
};

struct Scenario {
    long n_nodes;
    int node_size, line_bytes, sets, ways, mlp;
    double miss_latency;
};

// Fixed scenarios: each walks n_nodes elements spaced node_size bytes
// apart (a strided/pointer-chase-shaped access pattern) through a cache
// with the given geometry, producing a real, non-hardcoded miss count.
int main() {
    static const Scenario scenarios[] = {
        {256,  128, 64, 16, 4, 4, 200.0},
        {512,   64, 64, 32, 8, 8, 100.0},
        {128,  256, 64,  8, 2, 2, 300.0},
        {512,   32, 64, 16, 4, 4, 150.0},
        {1024, 128, 64, 32, 8, 4, 100.0},
    };

    for (const auto& sc : scenarios) {
        Level lvl(sc.sets, sc.ways);
        for (long i = 0; i < sc.n_nodes; i++) {
            long addr = i * (long)sc.node_size;
            lvl.access(addr / sc.line_bytes);
        }
        double cycles = modeled_cycles(lvl.misses, sc.mlp, sc.miss_latency);
        printf("misses=%ld cycles=%.6f\n", lvl.misses, cycles);
    }
    return 0;
}
