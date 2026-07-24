#include "sol.hpp"

// TODO: classify the twelve structs S1..S12 documented in task.md.
// For struct k (1-based): out[2*(k-1)]   = 1 if Sk is standard-layout, else 0
//                         out[2*(k-1)+1] = 1 if Sk is trivially-copyable, else 0
//
// Reason about each struct from the standard-layout and trivially-copyable
// rules and set the 24 bits. (You may instead paste the struct definitions
// here and query <type_traits> directly.)
void classify(int out[24]) {
    for (int i = 0; i < 24; i++) out[i] = 0;  // placeholder: all zeros -> wrong
}
