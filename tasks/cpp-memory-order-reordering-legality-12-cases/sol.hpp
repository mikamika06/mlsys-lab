#pragma once

enum class MemOrder { Relaxed, Acquire, Release, AcqRel, SeqCst, NonAtomic };
enum class OpType { Read, Write, Rmw };

struct MemOp {
    OpType type;
    int field_idx;
    MemOrder order;
};

struct ReorderCase {
    MemOp op1; // program-order-first
    MemOp op2; // program-order-second
};

// For each of the n cases, decide whether the C++ memory model PERMITS
// reordering op1 past op2 (i.e. the compiler/hardware may execute op2
// before op1 becomes visible):
//   - same field_idx on both ops        -> forbidden (0): true dependency.
//   - either op carries memory_order_seq_cst -> forbidden (0).
//   - op1 is acquire or acq_rel         -> forbidden (0): nothing may move
//                                          before an acquire.
//   - op2 is release or acq_rel         -> forbidden (0): nothing may move
//                                          after a release.
//   - otherwise                         -> permitted (1).
// Write out[i] = 1 (permitted) or 0 (forbidden) for cases[i], for i in
// [0, n).
void predict_reordering_legality(const ReorderCase* cases, int n, int* out);
