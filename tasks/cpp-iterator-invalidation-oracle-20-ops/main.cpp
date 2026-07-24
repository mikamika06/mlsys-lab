#include <cstdio>
#include "sol.hpp"

int main() {
    IterScenario s[20] = {
        // vector
        {CONTAINER_VECTOR, OP_PUSH_BACK, 5, 5,  2, 5},   // 0: size+1(6)>cap(5) -> realloc -> 0
        {CONTAINER_VECTOR, OP_PUSH_BACK, 3, 10, 1, 3},   // 1: no realloc, iter(1)<op(3) -> 1
        {CONTAINER_VECTOR, OP_INSERT,    5, 10, 3, 2},   // 2: no realloc, iter(3)>=op(2) -> 0
        {CONTAINER_VECTOR, OP_INSERT,    5, 10, 1, 3},   // 3: no realloc, iter(1)<op(3) -> 1
        {CONTAINER_VECTOR, OP_ERASE,     6, 10, 4, 2},   // 4: iter(4)>=op(2) -> 0
        {CONTAINER_VECTOR, OP_ERASE,     6, 10, 1, 3},   // 5: iter(1)<op(3) -> 1
        {CONTAINER_VECTOR, OP_RESERVE,   3, 5,  1, 10},  // 6: N(10)>cap(5) -> realloc -> 0
        {CONTAINER_VECTOR, OP_RESERVE,   3, 10, 1, 5},   // 7: N(5)<=cap(10) -> 1
        {CONTAINER_VECTOR, OP_INSERT,    4, 10, 2, 2},   // 8: iter(2)>=op(2), boundary -> 0
        // deque
        {CONTAINER_DEQUE, OP_PUSH_BACK, 4, 0, 2, 4},     // 9: always -> 0
        {CONTAINER_DEQUE, OP_ERASE,     4, 0, 0, 2},     // 10: always -> 0
        {CONTAINER_DEQUE, OP_INSERT,    4, 0, 3, 1},     // 11: always -> 0
        // list
        {CONTAINER_LIST, OP_PUSH_BACK, 4, 0, 2, 4},      // 12: -> 1
        {CONTAINER_LIST, OP_INSERT,    4, 0, 1, 2},      // 13: -> 1
        {CONTAINER_LIST, OP_ERASE,     5, 0, 3, 3},      // 14: iter==op -> 0
        {CONTAINER_LIST, OP_ERASE,     5, 0, 1, 3},      // 15: iter!=op -> 1
        // map
        {CONTAINER_MAP, OP_INSERT,    4, 0, 2, 1},       // 16: -> 1
        {CONTAINER_MAP, OP_PUSH_BACK, 4, 0, 0, 4},       // 17: -> 1
        {CONTAINER_MAP, OP_ERASE,     5, 0, 2, 2},       // 18: iter==op -> 0
        {CONTAINER_MAP, OP_ERASE,     5, 0, 4, 2},       // 19: iter!=op -> 1
    };

    for (int i = 0; i < 20; i++) printf("%d ", classify_iterator_validity(s[i]));
    printf("\n");
    return 0;
}
