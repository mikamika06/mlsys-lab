#include <cstdio>
#include "sol.hpp"

static int g_incref_count = 0;
static int g_decref_count = 0;

void incref(PyObj* obj) {
    obj->refcount++;
    g_incref_count++;
}
void decref(PyObj* obj) {
    obj->refcount--;
    g_decref_count++;
}

static void run_case(bool increment_input_refs) {
    PyObj a{1, 10};
    PyObj b{1, 20};
    PyObj c{1, 30};
    PyObj* items[3] = {&a, &b, &c};
    int before[3] = {a.refcount, b.refcount, c.refcount};

    g_incref_count = 0;
    g_decref_count = 0;

    PyTuple* t = make_tuple(items, 3, increment_input_refs);

    int after[3] = {a.refcount, b.refcount, c.refcount};

    printf("increment_input_refs=%d\n", increment_input_refs ? 1 : 0);
    printf("deltas=%d,%d,%d\n", after[0] - before[0], after[1] - before[1], after[2] - before[2]);
    if (t == nullptr) {
        printf("tuple=null\n");
    } else {
        int items_ok = (t->n == 3 && t->items[0] == &a && t->items[1] == &b && t->items[2] == &c) ? 1 : 0;
        printf("tuple_refcount=%d tuple_n=%d items_ok=%d\n", t->refcount, t->n, items_ok);
    }
    printf("incref_count=%d decref_count=%d\n", g_incref_count, g_decref_count);
}

// FIXED driver. Runs the "stolen reference" (increment_input_refs=false)
// and "fresh reference" (increment_input_refs=true) scenarios in turn.
int main() {
    run_case(true);
    run_case(false);
    return 0;
}
