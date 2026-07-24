#include "sol.hpp"

static bool is_seq_cst(MemOrder o) { return o == MemOrder::SeqCst; }
static bool is_acquire_ish(MemOrder o) { return o == MemOrder::Acquire || o == MemOrder::AcqRel; }
static bool is_release_ish(MemOrder o) { return o == MemOrder::Release || o == MemOrder::AcqRel; }

void predict_reordering_legality(const ReorderCase* cases, int n, int* out) {
    for (int i = 0; i < n; i++) {
        const MemOp& op1 = cases[i].op1;
        const MemOp& op2 = cases[i].op2;

        if (op1.field_idx == op2.field_idx) {
            out[i] = 0; // true dependency: same location
            continue;
        }
        if (is_seq_cst(op1.order) || is_seq_cst(op2.order)) {
            out[i] = 0;
            continue;
        }
        if (is_acquire_ish(op1.order)) {
            out[i] = 0; // nothing may move before an acquire
            continue;
        }
        if (is_release_ish(op2.order)) {
            out[i] = 0; // nothing may move after a release
            continue;
        }
        out[i] = 1;
    }
}
