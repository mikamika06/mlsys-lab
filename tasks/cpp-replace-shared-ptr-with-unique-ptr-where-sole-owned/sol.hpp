#pragma once
#include <string>

// Real, measured facts about ONE managed type T, gathered by main.cpp from
// actual std::unique_ptr<T>/std::shared_ptr<T> instances -- never a
// hand-rolled ABI table.
struct TypeFacts {
    int object_bytes;      // real sizeof(T)
    int unique_ptr_bytes;  // real sizeof(std::unique_ptr<T>)
    int shared_ptr_bytes;  // real sizeof(std::shared_ptr<T>)
};

struct OwnershipPlan {
    std::string pointer_type;  // "unique_ptr" or "shared_ptr"
    int atomic_ops;
    int pointer_bytes;
    int control_block_bytes;
    int object_bytes;
};

// Choose std::unique_ptr for sole ownership (0 atomic ops, its real
// pointer size, no control block) or std::shared_ptr for shared ownership
// (its real pointer size, a 16-byte control block -- the strong and weak
// atomic reference counters -- and 2 atomic ops per ownership transfer:
// one increment on copy, one decrement on the copy's destruction; main.cpp
// verifies that model against a real shared_ptr's use_count()).
OwnershipPlan optimize_ownership(bool is_sole_owned, int transfers, const TypeFacts& facts);
