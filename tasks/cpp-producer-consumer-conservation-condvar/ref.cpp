#include "sol.hpp"

void bq_init(BoundedQueue& q, int capacity) {
    q.head = 0;
    q.count = 0;
    q.capacity = capacity;
}

void bq_push(BoundedQueue& q, Item item, double* produced_sum) {
    std::lock_guard<std::mutex> lock(q.mtx);
    if (q.count >= q.capacity) {
        return; // full: drop
    }
    int tail = (q.head + q.count) % 8;
    q.items[tail] = item;
    q.count++;
    *produced_sum += item.payload;
}

void bq_pop(BoundedQueue& q, double* consumed_sum) {
    std::lock_guard<std::mutex> lock(q.mtx);
    if (q.count == 0) {
        return; // empty: drop
    }
    Item front = q.items[q.head];
    q.head = (q.head + 1) % 8;
    q.count--;
    *consumed_sum += front.payload;
}
