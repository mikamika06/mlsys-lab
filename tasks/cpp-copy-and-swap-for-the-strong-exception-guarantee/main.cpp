#include <cstdio>
#include <exception>
#include "sol.hpp"

static void print_buf(const ByteBuffer& b) {
    for (int i = 0; i < b.size; i++) printf("%d ", (int)b.data[i]);
    printf("\n");
}

// FIXED driver. Two scenarios:
//   1. A normal assignment (source not poisoned) -> target must become the
//      source's bytes.
//   2. An assignment where copying the source THROWS (source poisoned) ->
//      target must be left byte-identical to what it was BEFORE the
//      assignment (the strong exception guarantee).
int main() {
    const unsigned char t0[4] = {1, 2, 3, 4};
    const unsigned char s0[4] = {5, 6, 7, 8};

    {
        ByteBuffer target(t0, 4, /*is_poisoned=*/false);
        ByteBuffer source(s0, 4, /*is_poisoned=*/false);
        target = source;
        print_buf(target);
    }

    {
        ByteBuffer target(t0, 4, /*is_poisoned=*/false);
        ByteBuffer source(s0, 4, /*is_poisoned=*/true);
        bool threw = false;
        try {
            target = source;
        } catch (const std::exception&) {
            threw = true;
        }
        printf("threw=%d\n", threw ? 1 : 0);
        print_buf(target);
    }

    return 0;
}
