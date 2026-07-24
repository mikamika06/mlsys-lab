#include <new>
#include "sol.hpp"

int slab_reuse_demo() {
    alignas(Slot) unsigned char storage[sizeof(Slot)];

    Slot* p1 = new (storage) Slot{5};
    (void)p1;

    // Reuse the same storage for a NEW Slot object: p1's object's lifetime
    // ends here, a new one begins at the same address.
    Slot* p2 = new (storage) Slot{11};

    // p2 is already a valid pointer to the new object (it's literally what
    // placement-new just returned). std::launder makes that explicit and is
    // the standards-mandated way to re-derive a valid pointer from the raw
    // storage address if you didn't keep p2 around.
    Slot* lp = std::launder(p2);
    int result = lp->value;

    lp->~Slot();
    return result;
}
