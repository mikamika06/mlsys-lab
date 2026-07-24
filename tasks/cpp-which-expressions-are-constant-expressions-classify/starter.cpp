#include "sol.hpp"

// TODO: Classify the 12 expressions in task.md.
// Return a 12-bit mask: set bit i (i = 0..11) to 1 iff expression (i + 1) is a
// constant expression (usable as an array bound / non-type template argument /
// static_assert operand). Bits 12..31 must stay 0.
//
// The stub below classifies nothing and fails the gate on purpose.
unsigned classify_constexpr() {
    return 0; // your classification here
}
