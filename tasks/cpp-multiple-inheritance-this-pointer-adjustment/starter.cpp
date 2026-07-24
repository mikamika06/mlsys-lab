#include "sol.hpp"

// TODO: fill offs[0..3) with the this-pointer adjustment for Derived* ->
// B1*, Derived* -> B2*, Derived* -> B3*, then offs[3] with
// sizeof(Derived). See sol.hpp for what "adjustment" means and why B2/B3
// are not simply 0.
void base_offsets(std::size_t offs[4]) {
    // your code here: a WRONG but common assumption is that casting a
    // pointer to any base never changes its numeric value.
    offs[0] = 0;
    offs[1] = 0;
    offs[2] = 0;
    offs[3] = 0;
}
