// FIXED driver. Builds a real intrusively-refcounted `Obj` (a minimal
// stand-in for a CPython PyObject's `ob_refcnt` field) and real analogues
// of the C-API refcounting primitives, runs the 12 documented operation
// sequences on a real object for each, and reads back the REAL resulting
// refcount as ground truth -- never a hardcoded table.
#include <cstdio>
#include <string>
#include <vector>

#include "sol.hpp"

struct Obj {
    long ob_refcnt;
};

static inline void Py_INCREF_(Obj* o) { o->ob_refcnt++; }
static inline void Py_DECREF_(Obj* o) { o->ob_refcnt--; }
static inline Obj* Py_NewRef_(Obj* o) { Py_INCREF_(o); return o; }
static inline Obj* Py_Borrow_(Obj* o) { return o; }  // no refcount change

static int run_sequence(const std::vector<std::string>& ops) {
    Obj obj{1};  // freshly created object: refcount starts at 1
    for (const auto& op : ops) {
        if (op == "New") {
            Py_NewRef_(&obj);
        } else if (op == "Incref") {
            Py_INCREF_(&obj);
        } else if (op == "Decref") {
            Py_DECREF_(&obj);
        } else if (op == "Borrow") {
            Py_Borrow_(&obj);
        }
    }
    return (int)obj.ob_refcnt;
}

int main() {
    const std::vector<std::vector<std::string>> sequences = {
        {"New"},
        {"Incref", "Decref"},
        {"Incref", "Incref", "Decref"},
        {"Borrow", "Incref"},
        {"Incref", "Borrow", "Decref", "Decref"},
        {"New", "Incref", "Borrow", "Decref", "Decref"},
        {"New", "New"},
        {"Borrow", "Borrow"},
        {"Incref", "Incref", "Incref", "Decref", "Decref", "Decref"},
        {"Decref"},
        {"Incref", "Decref", "Incref", "Decref", "Incref"},
        {"New", "Borrow", "Incref", "Decref", "New", "Decref"},
    };

    int truth[12];
    for (int i = 0; i < 12; i++) truth[i] = run_sequence(sequences[i]);

    int pred[12];
    predict_refcounts(pred);

    int matches = 0;
    for (int i = 0; i < 12; i++) {
        int ok = (pred[i] == truth[i]) ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
