#pragma once

enum Container { CONTAINER_VECTOR = 0, CONTAINER_DEQUE = 1, CONTAINER_LIST = 2, CONTAINER_MAP = 3 };
enum Operation { OP_PUSH_BACK = 0, OP_INSERT = 1, OP_ERASE = 2, OP_RESERVE = 3 };

// One container-mutation scenario to classify.
struct IterScenario {
    int container;    // Container enum
    int operation;     // Operation enum
    int size;           // current element count
    int capacity;        // current capacity (vector only)
    int iter_pos;         // 0-based position the held iterator points at
    int op_pos;            // 0-based insert/erase position; for OP_RESERVE
                            // this instead carries N, the requested new capacity
};

// Classify whether a held iterator remains valid (1) or is invalidated (0)
// by the described mutation, following the standard container-iterator
// invalidation rules:
//
//   vector, push_back / insert:
//     if size + 1 > capacity        -> reallocation -> 0 for EVERY iterator
//     else if iter_pos >= op_pos    -> 0 (shifted/at the insertion point)
//     else                          -> 1
//   vector, erase:
//     iter_pos >= op_pos            -> 0
//     else                          -> 1
//   vector, reserve (N = op_pos):
//     N > capacity                  -> 0 for EVERY iterator (reallocation)
//     else                          -> 1 (no reallocation, nothing moves)
//   deque, ANY operation:
//     always                        -> 0 (every mutation invalidates everything)
//   list / map, push_back / insert:
//     always                        -> 1 (existing iterators never move)
//   list / map, erase:
//     iter_pos == op_pos            -> 0 (only the erased element's own iterator dies)
//     else                          -> 1
int classify_iterator_validity(const IterScenario& s);
