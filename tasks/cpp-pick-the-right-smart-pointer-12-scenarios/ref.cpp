#include "sol.hpp"

std::vector<std::string> smart_pointer_selection() {
    return {
        "unique",  //  1: exclusive ownership, automatic deletion
        "shared",  //  2: shared ownership until last subsystem dies
        "unique",  //  3: cache exclusively owns its entries
        "unique",  //  4: factory transfers sole ownership to caller
        "raw",     //  5: non-owning reference, lifetime guaranteed by caller
        "weak",    //  6: back-pointer in an owning graph, must not cycle
        "raw",     //  7: legacy C API, no ownership transfer
        "shared",  //  8: shared ownership across threads, freed after all finish
        "unique",  //  9: polymorphic deletion via virtual destructor, sole owner
        "raw",     // 10: hot loop, no refcount overhead
        "shared",  // 11: copyable pimpl handle sharing ownership of impl
        "weak",    // 12: non-owning back-pointer inside a shared_ptr graph
    };
}
