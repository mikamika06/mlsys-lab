#pragma once
#include <vector>

// Instrumented counters, DEFINED in main.cpp and incremented by Buffer's own
// copy/move special members (also defined in main.cpp -- the instrumentation
// is not your job, only DRIVING which operation actually happens is).
extern int g_copy_count;
extern int g_move_count;

// A movable, copyable owning buffer. Its copy ctor/assign bump
// g_copy_count; its (noexcept) move ctor/assign bump g_move_count -- exactly
// what a profiler hooked into these special members would observe. Because
// the move constructor is noexcept, std::vector<Buffer> uses it (not a
// copy) when reallocating.
struct Buffer {
    int id;
    double* data;
    long capacity;
    long size;

    Buffer();
    Buffer(const Buffer& other);
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(const Buffer& other);
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer();
};

// One pipeline instruction.
//   kind == 0  push_temp     ->  vec.push_back(Buffer());              (moves a temporary)
//   kind == 1  push_lvalue   ->  Buffer b; vec.push_back(b);            (copies an lvalue)
//   kind == 2  copy_assign   ->  vec[dst] = vec[src];                  (copy assignment)
//   kind == 3  move_assign   ->  vec[dst] = std::move(vec[src]);       (move assignment)
// `dst`/`src` are only meaningful for kind 2/3 and index already-pushed
// elements of `vec`.
struct Op {
    int kind;
    int dst;
    int src;
};

// Run ops[0..n) against `vec` (starts empty) IN ORDER, performing exactly
// the operation each op names above. Any reallocation std::vector performs
// along the way happens naturally and is counted by the same instrumented
// special members -- you don't count anything yourself, you just have to
// perform the RIGHT C++ operation for each op (the right value category:
// an rvalue temporary vs. a named lvalue vs. an explicit std::move).
void run_pipeline(const Op* ops, int n, std::vector<Buffer>& vec);
