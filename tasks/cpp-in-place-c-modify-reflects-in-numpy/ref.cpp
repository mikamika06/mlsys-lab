#include "sol.hpp"

void mutate_a(RecordA* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i].d1 += 1.0;
        arr[i].d2 += 1.0;
    }
}

void mutate_b(RecordB* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i].d += 1.0;
        arr[i].d2 += 1.0;
    }
}

void mutate_c(RecordC* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i].d += 1.0;
        arr[i].d2 += 1.0;
    }
}
