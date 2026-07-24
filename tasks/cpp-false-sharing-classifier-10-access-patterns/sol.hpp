#pragma once
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// PROVIDED (do not change): the real, compiler-laid-out struct. Assume an
// instance is allocated at an address aligned to 64 bytes (a real cache
// line size), so byte offset i falls in cache line i / 64.
// ---------------------------------------------------------------------------
struct ThreadState {
    int       id;
    int       read_count;
    long      write_count;
    double    items[5];
    long long padding[5];
    double    local_sum;
    int       local_flag;
};

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Classify 10 fixed concurrent access patterns on ONE ThreadState instance
// shared by two threads. A pattern suffers FALSE SHARING exactly when both
// accesses fall in the SAME 64-byte cache line AND at least one of the two
// is a write (two simultaneous reads never cause false sharing; accesses
// to different lines never do either, no matter how many are writes).
//
//    1. T1 writes id,          T2 writes read_count
//    2. T1 writes id,          T2 reads  write_count
//    3. T1 reads  read_count,  T2 reads  write_count
//    4. T1 writes items[4],    T2 writes padding[0]
//    5. T1 writes items[4],    T2 writes padding[1]
//    6. T1 writes padding[0],  T2 writes padding[1]
//    7. T1 writes local_sum,   T2 writes local_flag
//    8. T1 writes items[0],    T2 reads  local_sum
//    9. T1 writes padding[1],  T2 reads  local_flag
//   10. T1 reads  items[0],    T2 writes items[1]
//
// Return {labels, struct_size}: `labels` is exactly 10 booleans in the
// order above (true = false sharing), and `struct_size` is
// sizeof(ThreadState) as the real compiler lays it out -- use `offsetof`
// on the real struct to find each field's byte offset, do not guess it.
// ---------------------------------------------------------------------------
std::pair<std::vector<bool>, long> classify_false_sharing();
