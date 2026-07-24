#include <cstdio>
#include "sol.hpp"

// PROVIDED. Element's real constructors, and the global move counter.
Element::Element() : type(0), data(nullptr), sizes{0, 0, 0} {}

Element::Element(Element&& other) noexcept
    : type(other.type), data(other.data),
      sizes{other.sizes[0], other.sizes[1], other.sizes[2]} {
    ++g_move_count;
}

long g_move_count = 0;

// FIXED driver. Do not edit. Runs the learner's simulator over four fixed
// (N, initial_capacity, growth_factor) cases, printing sizeof(Element)
// once and, per case, the reallocation count and the REAL move count
// observed via Element's own move constructor.
int main() {
    struct Case { int N, init_cap, growth; };
    Case cases[] = {{10, 0, 2}, {100, 0, 2}, {50, 5, 2}, {1000, 10, 2}};

    printf("sizeof=%d\n", (int)sizeof(Element));
    for (const auto& c : cases) {
        g_move_count = 0;
        long reallocs = simulate_vector_pushes(c.N, c.init_cap, c.growth);
        printf("N=%d cap0=%d k=%d reallocs=%ld moves=%ld\n",
               c.N, c.init_cap, c.growth, reallocs, g_move_count);
    }
    return 0;
}
