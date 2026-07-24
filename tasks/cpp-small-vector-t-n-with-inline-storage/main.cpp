#include <cstdio>
#include "sol.hpp"

// FIXED driver. Builds a deterministic sequence that overflows the inline
// buffer (CAP = 4) so the vector must spill to the heap, then checks that
// lifetime bookkeeping came out right. Prints only integers.
int main() {
    Tracked::alive = 0;
    Tracked::ctor_calls = 0;
    Tracked::dtor_calls = 0;

    long checksum = 0;
    int  final_size = 0;
    int  did_spill = 0;
    long alive_during = 0;

    {
        SmallVector v;
        const long vals[] = {3, 1, 4, 1, 5, 9, 2, 6};  // 8 values, CAP = 4
        for (long x : vals) v.push_back(x);

        final_size   = v.size();          // expected 8
        checksum     = v.sum();           // expected 31
        did_spill    = v.spilled() ? 1 : 0;   // expected 1
        alive_during = Tracked::alive;    // expected 8 (all elements live)
    }  // v destroyed here

    long alive_after   = Tracked::alive;                                   // expected 0
    int  ctor_eq_dtor  = (Tracked::ctor_calls == Tracked::dtor_calls) ? 1 : 0; // expected 1
    long leaked        = Tracked::ctor_calls - Tracked::dtor_calls;        // expected 0

    printf("size=%d\n", final_size);
    printf("sum=%ld\n", checksum);
    printf("spilled=%d\n", did_spill);
    printf("alive_during=%ld\n", alive_during);
    printf("alive_after=%ld\n", alive_after);
    printf("ctor_eq_dtor=%d\n", ctor_eq_dtor);
    printf("leaked=%ld\n", leaked);
    return 0;
}
