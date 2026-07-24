#include "sol.hpp"
#include <cstdio>

int main() {
    using MO = MemOrder;
    using OT = OpType;

    static const ReorderCase cases[12] = {
        // 0: op1 relaxed write, op2 relaxed read (diff fields) -> 1
        {{OT::Write, 0, MO::Relaxed}, {OT::Read, 1, MO::Relaxed}},
        // 1: op1 acquire read, op2 relaxed write -> 0
        {{OT::Read, 0, MO::Acquire}, {OT::Write, 1, MO::Relaxed}},
        // 2: op1 relaxed write, op2 release write -> 0
        {{OT::Write, 0, MO::Relaxed}, {OT::Write, 1, MO::Release}},
        // 3: op1 release write, op2 acquire read -> 1
        {{OT::Write, 0, MO::Release}, {OT::Read, 1, MO::Acquire}},
        // 4: same field read/write -> 0
        {{OT::Read, 0, MO::Relaxed}, {OT::Write, 0, MO::Relaxed}},
        // 5: op1 seq_cst write -> 0
        {{OT::Write, 0, MO::SeqCst}, {OT::Read, 1, MO::Relaxed}},
        // 6: op2 seq_cst write -> 0
        {{OT::Read, 0, MO::Relaxed}, {OT::Write, 1, MO::SeqCst}},
        // 7: both relaxed write -> 1
        {{OT::Write, 0, MO::Relaxed}, {OT::Write, 1, MO::Relaxed}},
        // 8: both relaxed read -> 1
        {{OT::Read, 0, MO::Relaxed}, {OT::Read, 1, MO::Relaxed}},
        // 9: op1 acq_rel rmw -> 0
        {{OT::Rmw, 0, MO::AcqRel}, {OT::Write, 1, MO::Relaxed}},
        // 10: op1 release write, op2 relaxed write -> 1
        {{OT::Write, 0, MO::Release}, {OT::Write, 1, MO::Relaxed}},
        // 11: op1 relaxed read, op2 acquire read -> 1
        {{OT::Read, 0, MO::Relaxed}, {OT::Read, 1, MO::Acquire}},
    };

    int out[12];
    predict_reordering_legality(cases, 12, out);
    for (int i = 0; i < 12; i++) {
        printf("%d\n", out[i]);
    }
    return 0;
}
