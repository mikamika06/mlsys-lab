#pragma once
#include <mutex>

struct Item {
    int key;
    double payload;
};

struct BoundedQueue {
    std::mutex mtx;
    Item items[8]; // capacity is always <= 8 in this task's fixed cases
    int head = 0;
    int count = 0;
    int capacity = 0;
};

void bq_init(BoundedQueue& q, int capacity);

// Push `item` onto the back of the queue. Must lock q.mtx for the whole
// critical section (std::lock_guard is the natural tool). If the queue is
// already at capacity, drop the push (do nothing) — never block, never
// exceed `capacity`. On a successful push, add item.payload to
// *produced_sum.
void bq_push(BoundedQueue& q, Item item, double* produced_sum);

// Pop the item at the front of the queue (FIFO order). Must lock q.mtx
// for the whole critical section. If the queue is empty, drop the pop (do
// nothing) — never block. On a successful pop, add the popped item's
// payload to *consumed_sum.
void bq_pop(BoundedQueue& q, double* consumed_sum);
