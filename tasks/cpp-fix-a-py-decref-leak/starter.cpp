#include "sol.hpp"

// BROKEN: acquires a new reference every iteration but forgets to release
// it on the error path — a classic Py_DECREF leak on early return.
int process_items(MyPyObject* objs, int n) {
    for (int i = 0; i < n; ++i) {
        MyPyObject* obj = &objs[i];
        obj_incref(obj);          // acquire a new/owned reference

        if (obj->value < 0) {
            // BUG: missing obj_decref(obj) before returning — the acquired
            // reference leaks.
            return -1;
        }

        obj_decref(obj);          // success path: release, keep going
    }
    return 0;
}
