#include "sol.hpp"
#include <new>

// TODO: implement SmallVector with inline storage for CAP elements, spilling to
// the heap once more than CAP elements are pushed. Use placement new to
// construct each element and manual destructor calls to destroy it — construct
// and destroy every element exactly once (no leaks, no double-free).

SmallVector::SmallVector()
    : data(reinterpret_cast<Tracked*>(inbuf)), sz(0), cap(CAP) {
    // your code here
}

SmallVector::~SmallVector() {
    // your code here
}

void SmallVector::push_back(long v) {
    (void)v;  // your code here
}

long SmallVector::sum() const {
    return 0;  // your code here
}

int SmallVector::size() const {
    return 0;  // your code here
}

bool SmallVector::spilled() const {
    return false;  // your code here
}
