#pragma once
#include <atomic>

// The role played by one std::atomic operation in a larger synchronization
// pattern -- this drives which std::memory_order is the WEAKEST one that
// still keeps the pattern correct under the real C++ memory model.
// Strength order: relaxed < acquire/release < acq_rel < seq_cst.
enum class OpRole {
    CounterIncrement,  // standalone counter++, nothing else depends on it
    PublishStore,       // writes a payload, THEN sets a "ready" flag
    ConsumeLoad,          // reads a "ready" flag, THEN reads the payload
    LockAcquire,           // acquires a spinlock (CAS / test-and-set)
    LockRelease,            // releases a spinlock (store / clear)
    RmwSync,                 // a read-modify-write that must both acquire AND
                               // release (e.g. a refcount decrement guarding a dtor)
    TotalOrder,                // needs a single global total order across ALL
                                 // threads (e.g. Dekker's algorithm / store-buffering)
    RelaxedRead,                 // reads a value with no synchronization requirement
};

// Returns the WEAKEST std::memory_order (the real enum from <atomic>) that
// keeps `role` correct:
//   CounterIncrement -> memory_order_relaxed
//   PublishStore      -> memory_order_release  (must happen-before the flag is seen)
//   ConsumeLoad       -> memory_order_acquire  (must happen-after the flag publishes)
//   LockAcquire        -> memory_order_acquire  (must see everything the last unlock released)
//   LockRelease         -> memory_order_release  (must publish everything done under the lock)
//   RmwSync              -> memory_order_acq_rel (the operation is both an acquire and a release)
//   TotalOrder            -> memory_order_seq_cst (needs the single global total order)
//   RelaxedRead            -> memory_order_relaxed
std::memory_order weakestOrderFor(OpRole role);
