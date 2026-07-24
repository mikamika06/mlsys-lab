#include "sol.hpp"

std::memory_order weakestOrderFor(OpRole role) {
    switch (role) {
        case OpRole::CounterIncrement: return std::memory_order_relaxed;
        case OpRole::PublishStore:     return std::memory_order_release;
        case OpRole::ConsumeLoad:      return std::memory_order_acquire;
        case OpRole::LockAcquire:      return std::memory_order_acquire;
        case OpRole::LockRelease:      return std::memory_order_release;
        case OpRole::RmwSync:          return std::memory_order_acq_rel;
        case OpRole::TotalOrder:       return std::memory_order_seq_cst;
        case OpRole::RelaxedRead:      return std::memory_order_relaxed;
    }
    return std::memory_order_seq_cst;
}
