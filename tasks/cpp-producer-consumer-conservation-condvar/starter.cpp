#include "sol.hpp"

void bq_init(BoundedQueue& q, int capacity) {
    q.head = 0;
    q.count = 0;
    q.capacity = capacity;
}

void bq_push(BoundedQueue& q, Item item, double* produced_sum) {
    // your code here
}

void bq_pop(BoundedQueue& q, double* consumed_sum) {
    // your code here
}
