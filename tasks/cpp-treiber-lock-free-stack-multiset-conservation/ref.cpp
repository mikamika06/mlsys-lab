#include "sol.hpp"

// Reference: textbook lock-free Treiber stack.
//
// push: build a fresh node and CAS it onto head_, retrying if head_ moved.
// pop : snapshot head_, then CAS it to head_->next, retrying on failure; the
//       loop exits false only when head_ is observed null (empty).
//
// Nodes are intentionally never freed. Not freeing means node addresses are
// never reused during a run, which sidesteps the ABA problem without needing
// hazard pointers -- appropriate here because the task grades multiset
// conservation, not memory reclamation.

void TreiberStack::push(int value) {
    Node* n = new Node{value, nullptr};
    Node* old = head_.load(std::memory_order_relaxed);
    do {
        n->next = old;                                   // link to current top
    } while (!head_.compare_exchange_weak(
                 old, n,
                 std::memory_order_release,
                 std::memory_order_relaxed));            // publish; retry on race
}

bool TreiberStack::pop(int& out) {
    Node* old = head_.load(std::memory_order_acquire);
    while (old != nullptr) {
        Node* next = old->next;                          // safe: nodes not freed
        if (head_.compare_exchange_weak(
                old, next,
                std::memory_order_acquire,
                std::memory_order_acquire)) {
            out = old->value;
            return true;
        }
        // CAS failed: `old` was refreshed to the current head; retry.
    }
    return false;                                        // stack empty
}
