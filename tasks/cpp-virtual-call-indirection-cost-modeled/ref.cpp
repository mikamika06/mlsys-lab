#include "sol.hpp"

long naive_virtual_loads(const int* /*obj_id*/, const int* /*slot*/, int n) {
    // Every call pays a vptr load and a vtable-slot load.
    return 2L * (long)n;
}

long cached_virtual_loads(const int* obj_id, const int* slot, int n) {
    long total = 0;
    for (int i = 0; i < n; ++i) {
        bool same_obj  = (i > 0) && (obj_id[i] == obj_id[i - 1]);
        bool same_slot = same_obj && (slot[i] == slot[i - 1]);
        if (!same_obj)  total += 1;   // vptr load
        if (!same_slot) total += 1;   // vtable-slot load
    }
    return total;
}

long devirtualized_loads(const int* /*obj_id*/, const int* /*slot*/, int /*n*/) {
    return 0;
}
