#include <new>
#include "sol.hpp"

int slab_reuse_demo() {
    alignas(Slot) unsigned char storage[sizeof(Slot)];

    Slot* p1 = new (storage) Slot{5};
    int stale = p1->value;   // BUG: read (and cache) the value BEFORE reuse

    new (storage) Slot{11};  // reuse the storage for a new object

    // BUG: return the value cached before the reuse instead of reading the
    // NEW object's value through a properly re-derived pointer.
    return stale;
}
