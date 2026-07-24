#include <cstdio>
#include "sol.hpp"

static const char* roleName(OpRole r) {
    switch (r) {
        case OpRole::CounterIncrement: return "counter_increment";
        case OpRole::PublishStore:     return "publish_store";
        case OpRole::ConsumeLoad:      return "consume_load";
        case OpRole::LockAcquire:      return "lock_acquire";
        case OpRole::LockRelease:      return "lock_release";
        case OpRole::RmwSync:          return "rmw_sync";
        case OpRole::TotalOrder:       return "total_order";
        case OpRole::RelaxedRead:      return "relaxed_read";
    }
    return "?";
}

int main() {
    OpRole roles[] = {
        OpRole::CounterIncrement, OpRole::PublishStore, OpRole::ConsumeLoad,
        OpRole::LockAcquire, OpRole::LockRelease, OpRole::RmwSync,
        OpRole::TotalOrder, OpRole::RelaxedRead,
    };
    for (OpRole r : roles) {
        std::memory_order order = weakestOrderFor(r);
        printf("%s %d\n", roleName(r), (int)order);
    }
    return 0;
}
