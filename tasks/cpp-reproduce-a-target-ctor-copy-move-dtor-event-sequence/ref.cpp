#include "sol.hpp"
#include <utility>

// C Y M D D D:
//   Probe p1(1)          -> C
//   Probe p2(p1)          -> Y   (copy)
//   Probe p3(move(p2))    -> M   (move)
//   inner scope ends      -> D D  (p3, then p2, reverse declaration order)
//   outer scope ends      -> D    (p1)
void reproduce_sequence() {
    Probe p1(1);
    {
        Probe p2(p1);
        Probe p3(std::move(p2));
        (void)p3;
    }
}
