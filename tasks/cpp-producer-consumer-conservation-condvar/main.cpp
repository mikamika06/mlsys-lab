// Fixed driver: scripted push/pop sequences (single-threaded, fully
// deterministic — no real OS threads, no timing) exercising a real
// mutex-protected bounded queue's capacity and FIFO conservation
// invariants.
#include "sol.hpp"
#include <cstdio>

static void run_case(int capacity, const char* ops, const Item* push_items) {
    // ops is a string of 'P' (push, consumes the next push_items entry)
    // and 'O' (pop). push_items has one entry per 'P' in ops.
    BoundedQueue q;
    bq_init(q, capacity);
    double produced = 0.0, consumed = 0.0;
    int pi = 0;
    for (const char* c = ops; *c; c++) {
        if (*c == 'P') {
            bq_push(q, push_items[pi], &produced);
            pi++;
        } else {
            bq_pop(q, &consumed);
        }
    }
    printf("%.6f %.6f %d\n", produced, consumed, q.count);
}

int main() {
    {
        // capacity 2: push,push,push(dropped,full),pop,push,pop,pop
        Item items[] = {{10, 1.5}, {20, 2.5}, {30, 3.5}, {30, 3.5}};
        run_case(2, "PPPOPOO", items);
    }
    {
        // capacity 1: push, push(dropped,full), pop
        Item items[] = {{100, 100.0}, {200, 200.0}};
        run_case(1, "PPO", items);
    }
    {
        // capacity 3: pop(dropped,empty), push, push, pop, pop
        Item items[] = {{5, 50.0}, {6, 60.0}};
        run_case(3, "OPPOO", items);
    }
    {
        // capacity 2: push, push, push(dropped,full), pop -- one item left
        Item items[] = {{1, 1.0}, {2, 2.0}, {3, 3.0}};
        run_case(2, "PPPO", items);
    }
    return 0;
}
