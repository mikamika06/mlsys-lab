#include "sol.hpp"

int classify_iterator_validity(const IterScenario& s) {
    switch (s.container) {
        case CONTAINER_VECTOR: {
            if (s.operation == OP_RESERVE) {
                int n = s.op_pos;
                return (n > s.capacity) ? 0 : 1;
            }
            if (s.operation == OP_PUSH_BACK || s.operation == OP_INSERT) {
                if (s.size + 1 > s.capacity) return 0;   // reallocation
                return (s.iter_pos >= s.op_pos) ? 0 : 1;
            }
            // OP_ERASE
            return (s.iter_pos >= s.op_pos) ? 0 : 1;
        }
        case CONTAINER_DEQUE:
            return 0;   // any mutation invalidates every iterator

        case CONTAINER_LIST:
        case CONTAINER_MAP: {
            if (s.operation == OP_ERASE)
                return (s.iter_pos == s.op_pos) ? 0 : 1;
            return 1;   // push_back / insert never invalidate existing iterators
        }
        default:
            return 0;
    }
}
