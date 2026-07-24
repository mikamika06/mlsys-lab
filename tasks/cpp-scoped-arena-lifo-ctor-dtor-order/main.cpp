#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. Builds a deterministic id sequence, runs the
// learner's scoped arena, then prints the emitted event log plus the arena's
// memory accounting. A correct LIFO teardown prints the ids reversed in the
// destruction line.
int main() {
    const int ids[] = {7, 3, 9, 2, 5, 8};
    const int n = (int)(sizeof(ids) / sizeof(ids[0]));

    long bytes = run_scoped_arena(ids, n);

    // Event log: "C7 C3 C9 C2 C5 C8 D8 D5 D2 D9 D3 D7 " for a correct arena.
    printf("%s\n", g_events.c_str());
    printf("arena_bytes=%ld object_size=%ld count=%d\n",
           bytes, (long)sizeof(Probe), n);
    return 0;
}
