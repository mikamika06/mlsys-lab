#include <cstdio>
#include <cstdlib>
#include <new>
#include "sol.hpp"

// Instrumented global allocator: counts every heap allocation the program
// makes (and its size) from this point on. Overriding the GLOBAL operator
// new/delete applies link-wide, so it also sees every allocation made
// inside std::make_shared / std::shared_ptr's own internals -- exactly what
// makes the 1-vs-2 allocation difference directly observable, for real, no
// simulation involved.
static long g_alloc_count = 0;
static long g_total_bytes = 0;

void* operator new(std::size_t sz) {
    g_alloc_count++;
    g_total_bytes += (long)sz;
    void* p = std::malloc(sz);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

// FIXED driver.
int main() {
    // Warm-up: run stdio once before measuring, so any one-time lazy
    // allocation printf's own buffering makes doesn't leak into the counts
    // below.
    printf("%s", "");

    g_alloc_count = 0;
    g_total_bytes = 0;
    auto p1 = make_payload(true, 7, 3.5, 'A');
    long ms_allocs = g_alloc_count, ms_bytes = g_total_bytes;

    g_alloc_count = 0;
    g_total_bytes = 0;
    auto p2 = make_payload(false, 7, 3.5, 'A');
    long sp_allocs = g_alloc_count, sp_bytes = g_total_bytes;

    printf("make_shared:    allocs=%ld bytes=%ld a=%d b=%.1f c=%c use_count=%ld\n",
           ms_allocs, ms_bytes, p1->a, p1->b, p1->c, (long)p1.use_count());
    printf("shared_ptr_new: allocs=%ld bytes=%ld a=%d b=%.1f c=%c use_count=%ld\n",
           sp_allocs, sp_bytes, p2->a, p2->b, p2->c, (long)p2.use_count());
    return 0;
}
