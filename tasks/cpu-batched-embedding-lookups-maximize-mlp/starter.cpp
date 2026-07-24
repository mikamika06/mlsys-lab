#include "sol.hpp"

// BUG: treats every access as if it were a serial pointer chase, giving
// each one its own wave. This is a VALID schedule (satisfies both rules)
// but wastes the embedding lookups' independence entirely -- far from
// optimal.
void schedule_embedding_workload() {
    for (int i = 0; i < TOTAL_ACCESSES; i++) {
        schedule_access(i, i);
    }
}
