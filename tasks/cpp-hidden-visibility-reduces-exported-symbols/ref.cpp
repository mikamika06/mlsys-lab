#include "sol.hpp"

int count_exported_symbols(bool global_hidden, const int* is_static, const int* attr, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (is_static[i]) continue;  // internal linkage: never exported

        bool exported;
        if (attr[i] == 0) exported = true;          // explicit default
        else if (attr[i] == 1) exported = false;     // explicit hidden
        else exported = !global_hidden;               // inherits global default

        if (exported) count++;
    }
    return count;
}
