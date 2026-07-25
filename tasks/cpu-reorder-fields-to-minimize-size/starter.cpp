#include "sol.hpp"

// TODO: define a struct with the same 7 field types as NaiveStruct, sorted
// so that no field ever needs padding before it, and return its size.
// See sol.hpp for the exact field list and rules.
size_t packed_struct_size() {
    // your code here
    return sizeof(NaiveStruct);
}
