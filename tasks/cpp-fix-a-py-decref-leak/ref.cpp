#include "sol.hpp"

int process_items(MyPyObject* objs, int n) {
    for (int i = 0; i < n; ++i) {
        MyPyObject* obj = &objs[i];
        obj_incref(obj);          // acquire a new/owned reference

        if (obj->value < 0) {
            obj_decref(obj);      // release it before the early return
            return -1;
        }

        obj_decref(obj);          // success path: release, keep going
    }
    return 0;
}
