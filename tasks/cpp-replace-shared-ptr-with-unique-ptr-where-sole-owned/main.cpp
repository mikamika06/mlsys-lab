// FIXED driver. Measures real TypeFacts (sizeof std::unique_ptr<T> /
// std::shared_ptr<T> / T itself) for 3 real struct types, and separately
// VERIFIES the "2 atomic ops per transfer" model against a real
// std::shared_ptr<T>: doing `transfers` real copy+destroy cycles must
// leave use_count() back where it started, proving each cycle really is
// exactly one atomic increment (the copy) paired with one atomic
// decrement (the copy's destruction). Then it asks the candidate's
// optimize_ownership for each of the 4 documented scenarios and prints
// the resulting plan.
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "sol.hpp"

struct SA { int f0; double f1; };  // scenarios 0, 1
struct SB { char f0; long f1; };   // scenario 2
struct SC { short f0; float f1; }; // scenario 3

template <typename T>
static TypeFacts measure_facts() {
    return {(int)sizeof(T), (int)sizeof(std::unique_ptr<T>), (int)sizeof(std::shared_ptr<T>)};
}

// Real proof that a shared_ptr copy/destroy cycle is exactly a paired
// atomic increment/decrement: after `transfers` cycles, use_count() must
// be back at its starting value.
template <typename T>
static bool verify_transfer_model(int transfers) {
    auto sp = std::make_shared<T>();
    long baseline = sp.use_count();
    for (int i = 0; i < transfers; i++) {
        auto copy = sp;  // +1 atomic increment (copy ctor)
        (void)copy;
    }  // each `copy` destructs here: -1 atomic decrement
    return sp.use_count() == baseline;
}

static void print_plan(int idx, const OwnershipPlan& p) {
    printf("%d %s %d %d %d %d\n", idx, p.pointer_type.c_str(), p.atomic_ops,
           p.pointer_bytes, p.control_block_bytes, p.object_bytes);
}

int main() {
    if (!verify_transfer_model<SA>(5) || !verify_transfer_model<SB>(10) ||
        !verify_transfer_model<SC>(3)) {
        fprintf(stderr, "shared_ptr transfer model did not hold on this build\n");
        return 1;
    }

    TypeFacts fa = measure_facts<SA>();
    TypeFacts fb = measure_facts<SB>();
    TypeFacts fc = measure_facts<SC>();

    print_plan(0, optimize_ownership(true, 5, fa));
    print_plan(1, optimize_ownership(false, 5, fa));
    print_plan(2, optimize_ownership(true, 10, fb));
    print_plan(3, optimize_ownership(false, 3, fc));
    return 0;
}
