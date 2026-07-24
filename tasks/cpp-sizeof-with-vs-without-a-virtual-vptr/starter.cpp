#include "sol.hpp"

// TODO: derive the polymorphic sizeof from plain_size/plain_align. See
// sol.hpp for the exact vptr-prepend rule.
long virtual_sizeof(long plain_size, long plain_align) {
    (void)plain_size;
    (void)plain_align;
    return 0;
}
