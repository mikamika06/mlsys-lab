#include "sol.hpp"

double manual_dispatch(const Base* obj, int x) {
    // The vptr is the very first word of any polymorphic object; treat its
    // address as a pointer-to-(pointer-to-(function-pointer array)).
    void** vptr = *reinterpret_cast<void***>(const_cast<Base*>(obj));
    // Slot 0 holds compute's function pointer (declared first in Base).
    using ComputeFn = double (*)(const Base*, int);
    ComputeFn fn = reinterpret_cast<ComputeFn>(vptr[0]);
    return fn(obj, x);
}
