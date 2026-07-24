#include "sol.hpp"

// TODO: implement a lock-free Treiber stack using an atomic CAS loop on head_
// so that pop() returns exactly the multiset of values passed to push(), for
// ANY interleaving of producer and consumer threads (no lost / duplicated
// items).
//
// These stubs COMPILE but are wrong: push drops the value and pop never returns
// anything, so the driver sees an empty stack and the multiset check fails.

void TreiberStack::push(int value) {
    (void)value;
    // TODO: allocate a Node, link its next to the current head_,
    //       then CAS it onto head_ (retry while the CAS fails).
}

bool TreiberStack::pop(int& out) {
    (void)out;
    // TODO: load head_; if null return false; otherwise CAS head_ to
    //       head_->next and return the popped node's value.
    return false;
}
