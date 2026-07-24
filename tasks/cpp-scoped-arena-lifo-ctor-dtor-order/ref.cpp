#include "sol.hpp"
#include <new>       // placement new
#include <cstddef>   // std::size_t

// Correct reference: one contiguous raw arena, placement-new forward,
// explicit destructor calls in reverse (LIFO) order.
long run_scoped_arena(const int* ids, int n) {
    if (n <= 0) return 0;

    const std::size_t S = sizeof(Probe);
    // operator new returns storage aligned for any fundamental-aligned object,
    // so it is suitably aligned for Probe (alignof == 8).
    void*  raw   = ::operator new(S * (std::size_t)n);
    Probe* slots = static_cast<Probe*>(raw);

    // Construct in forward order.
    for (int i = 0; i < n; ++i)
        ::new (static_cast<void*>(slots + i)) Probe(ids[i]);

    // Destroy in strict LIFO / reverse-construction order.
    for (int i = n - 1; i >= 0; --i)
        (slots + i)->~Probe();

    ::operator delete(raw);
    return (long)(S * (std::size_t)n);
}
