#pragma once
#include <atomic>

// Lock-free (Treiber) stack of ints.
//
// Under concurrent producers (push) and consumers (pop), the multiset of values
// returned by pop() must equal the multiset of values passed to push(): no
// element may be lost, duplicated, or corrupted, for ANY thread interleaving.
//
// Implement push/pop with an atomic compare-and-swap (CAS) loop on head_:
//   push : allocate a node, point its next at the current head, then CAS it in;
//          retry if another thread changed head first.
//   pop  : read the current head; if null the stack is empty; otherwise CAS
//          head to head->next and return the popped node's value; retry on CAS
//          failure.
//
// (You may leak popped nodes -- safe memory reclamation for a lock-free stack is
// a separate hard problem and is NOT what this task grades.)
class TreiberStack {
public:
    TreiberStack() = default;

    // Push value onto the stack. Thread-safe and lock-free (CAS loop).
    void push(int value);

    // Pop the top value. If the stack is non-empty, store it in `out` and
    // return true; if empty, return false. Thread-safe and lock-free.
    bool pop(int& out);

private:
    struct Node {
        int   value;
        Node* next;
    };
    std::atomic<Node*> head_{nullptr};
};
