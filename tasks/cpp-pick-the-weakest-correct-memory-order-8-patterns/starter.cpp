#include "sol.hpp"

// BUG: acquire and release are swapped on the four roles where direction
// matters. A "publish" store needs RELEASE (so earlier writes happen-before
// it), not acquire; a "consume" load needs ACQUIRE (so later reads
// happen-after it), not release -- and the same swap is repeated for the
// lock roles.
std::memory_order weakestOrderFor(OpRole role) {
    switch (role) {
        case OpRole::CounterIncrement: return std::memory_order_relaxed;
        case OpRole::PublishStore:     return std::memory_order_acquire;   // wrong: should be release
        case OpRole::ConsumeLoad:      return std::memory_order_release;   // wrong: should be acquire
        case OpRole::LockAcquire:      return std::memory_order_release;   // wrong: should be acquire
        case OpRole::LockRelease:      return std::memory_order_acquire;   // wrong: should be release
        case OpRole::RmwSync:          return std::memory_order_acq_rel;
        case OpRole::TotalOrder:       return std::memory_order_seq_cst;
        case OpRole::RelaxedRead:      return std::memory_order_relaxed;
    }
    return std::memory_order_seq_cst;
}
