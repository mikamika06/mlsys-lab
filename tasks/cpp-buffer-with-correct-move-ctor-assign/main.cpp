#include <cstdio>
#include <utility>
#include "sol.hpp"

// ---- instrumented fake heap -------------------------------------------
namespace {
long g_next_id = 1;
long g_allocs = 0, g_frees = 0, g_deep_copies = 0;
bool g_live[64] = {};
}  // namespace

long tracked_alloc(long size) {
    (void)size;
    long id = g_next_id++;
    g_live[id] = true;
    g_allocs++;
    return id;
}

void tracked_free(long id) {
    if (id == 0) return;
    g_live[id] = false;
    g_frees++;
}

void tracked_deep_copy(long dst_id, long src_id, long size) {
    (void)dst_id; (void)src_id; (void)size;
    g_deep_copies++;
}

long stats_allocs() { return g_allocs; }
long stats_frees() { return g_frees; }
long stats_deep_copies() { return g_deep_copies; }

// ---- fixed scenario: construct, clone, move, copy-assign, move-assign,
// self-assign (both forms), then let RAII destroy everything at scope exit.
static void report(const char* tag) {
    printf("%s allocs=%ld frees=%ld deep=%ld\n", tag, stats_allocs(), stats_frees(), stats_deep_copies());
}

int main() {
    {
        Buffer b1(100);
        report("after_b1_ctor");
        printf("b1_null=%d b1_size=%ld\n", b1.ptr == 0, b1.size);

        Buffer b2(b1);  // copy ctor
        report("after_b2_clone");
        printf("b2_null=%d b2_size=%ld b1_null=%d b1_size=%ld\n",
               b2.ptr == 0, b2.size, b1.ptr == 0, b1.size);

        Buffer b3(std::move(b1));  // move ctor
        report("after_b3_move_ctor");
        printf("b3_null=%d b3_size=%ld b1_null=%d b1_size=%ld\n",
               b3.ptr == 0, b3.size, b1.ptr == 0, b1.size);

        Buffer b4(250);
        report("after_b4_ctor");

        b2 = b4;  // copy assign
        report("after_b2_copy_assign");
        printf("b2_null=%d b2_size=%ld b4_null=%d b4_size=%ld\n",
               b2.ptr == 0, b2.size, b4.ptr == 0, b4.size);

        Buffer b5(64);
        report("after_b5_ctor");

        b5 = std::move(b4);  // move assign
        report("after_b5_move_assign");
        printf("b5_null=%d b5_size=%ld b4_null=%d b4_size=%ld\n",
               b5.ptr == 0, b5.size, b4.ptr == 0, b4.size);

        Buffer* p5 = &b5;
        b5 = *p5;  // self copy-assign (must be a no-op)
        report("after_self_copy_assign");
        printf("b5_null=%d b5_size=%ld\n", b5.ptr == 0, b5.size);

        b5 = std::move(*p5);  // self move-assign (must be a no-op)
        report("after_self_move_assign");
        printf("b5_null=%d b5_size=%ld\n", b5.ptr == 0, b5.size);
    }  // b1..b5 destructors run here
    report("after_scope_exit");
    return 0;
}
