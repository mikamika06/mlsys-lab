#include "sol.hpp"
#include <vector>

namespace {

template <int Bytes, bool MoveNoexcept>
struct Elem {
    unsigned char payload[Bytes];
    Elem() {
        for (int i = 0; i < Bytes; i++) payload[i] = 0;
    }
    Elem(const Elem& o) {
        for (int i = 0; i < Bytes; i++) payload[i] = o.payload[i];
        g_counters.copies++;
    }
    Elem(Elem&& o) noexcept(MoveNoexcept) {
        for (int i = 0; i < Bytes; i++) payload[i] = o.payload[i];
        g_counters.moves++;
    }
    ~Elem() { g_counters.destructions++; }
};

template <int Bytes, bool MoveNoexcept>
GrowthCounts run(int n_pushes) {
    g_counters = Counters{};
    std::vector<Elem<Bytes, MoveNoexcept>> v;
    long last_cap = 0;
    long total_alloc = 0;
    for (int i = 0; i < n_pushes; i++) {
        v.emplace_back();
        long cap = static_cast<long>(v.capacity());
        if (cap != last_cap) {
            total_alloc += cap * static_cast<long>(sizeof(Elem<Bytes, MoveNoexcept>));
            last_cap = cap;
        }
    }
    GrowthCounts c;
    c.copies = g_counters.copies;
    c.moves = g_counters.moves;
    c.destructions = g_counters.destructions;
    c.total_alloc_bytes = total_alloc;
    c.final_capacity = static_cast<long>(v.capacity());
    return c;
}

} // namespace

GrowthCounts simulate_vector_growth(int element_size, int n_pushes, bool move_is_noexcept) {
    if (element_size == 16) {
        return move_is_noexcept ? run<16, true>(n_pushes) : run<16, false>(n_pushes);
    }
    return move_is_noexcept ? run<8, true>(n_pushes) : run<8, false>(n_pushes);
}
