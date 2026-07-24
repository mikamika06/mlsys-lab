#include "sol.hpp"

// Reference implementation: build a REAL std::vector, take a real pointer to the
// element at refIdx, replay the operations, and decide validity from what actually
// happens at runtime.
//
//  * Any reallocation (the storage pointer changes) invalidates every reference.
//  * Without reallocation, the original element survives only if it still lives
//    at its slot -- i.e. it was not erased (pop_back/clear reaching it) and not
//    displaced by an insertion at or before its index. Distinct sentinel values
//    let us detect displacement by identity.
bool ref_survives(int n0, int cap0, int refIdx, const std::vector<Op>& ops) {
    std::vector<long> v;
    v.reserve(cap0);                       // capacity becomes exactly cap0
    for (int i = 0; i < n0; i++) v.push_back(1000 + i);  // distinct originals

    const long* data0 = v.data();
    long origVal = v[refIdx];
    bool reallocated = false;
    long next = 5000;                      // sentinels disjoint from originals

    for (const Op& op : ops) {
        switch (op.kind) {
            case RESERVE:   v.reserve(op.arg); break;
            case PUSH_BACK: v.push_back(next++); break;
            case POP_BACK:  if (!v.empty()) v.pop_back(); break;
            case INSERT: {
                int p = op.arg;
                if (p < 0) p = 0;
                if (p > (int)v.size()) p = (int)v.size();
                v.insert(v.begin() + p, next++);
                break;
            }
            case CLEAR:     v.clear(); break;
        }
        if (v.data() != data0) reallocated = true;
    }

    if (reallocated) return false;
    if (refIdx >= (int)v.size()) return false;   // erased away (pop_back/clear)
    return v.data()[refIdx] == origVal;          // same element still in place?
}
