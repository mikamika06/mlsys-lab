#include "sol.hpp"

int classify_layout(const Field* fields, int nfields) {
    if (nfields == 0) return 0;
    for (int i = 0; i < nfields; i++) {
        if (fields[i].is_array) return 1;  // any array field -> SoA
    }
    return 0;  // all plain scalar fields -> AoS
}
