// Fixed driver + mock CPython C-API. Deterministic: fixed list of 3 real
// PyObjects with refcnt == 1, no real interpreter, no timing.
#include "sol.hpp"
#include <cstdio>

namespace {
PyObject g_items[3];
void* const g_list_handle = reinterpret_cast<void*>(0x1000);
PyObject* g_tuple_slots[3] = {nullptr, nullptr, nullptr};
void* const g_tuple_handle = reinterpret_cast<void*>(0x2000);
} // namespace

PyObject* PyList_GetItem(void* /*list*/, int i) {
    return &g_items[i];
}

int PyList_Size(void* /*list*/) {
    return 3;
}

void* PyTuple_New(int /*size*/) {
    for (int i = 0; i < 3; i++) g_tuple_slots[i] = nullptr;
    return g_tuple_handle;
}

void PyTuple_SetItem(void* /*tup*/, int i, PyObject* item) {
    g_tuple_slots[i] = item;
}

int main() {
    for (int i = 0; i < 3; i++) {
        g_items[i].ob_refcnt = 1;
        g_items[i].ob_type = nullptr;
    }

    void* tup = process_list_to_tuple(g_list_handle);

    printf("%d\n", tup == g_tuple_handle ? 1 : 0);
    for (int i = 0; i < 3; i++) {
        printf("%d %ld\n", g_tuple_slots[i] == &g_items[i] ? 1 : 0, g_items[i].ob_refcnt);
    }
    return 0;
}
