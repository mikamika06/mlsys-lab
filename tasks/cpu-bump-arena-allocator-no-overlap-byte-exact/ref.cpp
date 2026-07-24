#include "sol.hpp"

static long g_cursor = 0;

void bump_reset() { g_cursor = 0; }

long bump_alloc(long size, long align, long arena_bytes) {
    long rem = g_cursor % align;
    long aligned = rem == 0 ? g_cursor : g_cursor + (align - rem);

    if (aligned + size > arena_bytes) {
        return -1;
    }

    long offset = aligned;
    g_cursor = aligned + size;
    return offset;
}
