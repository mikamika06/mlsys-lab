#include "sol.hpp"
#include <set>

// Reference: an explicit operational model of the store-buffering test.
//
// Locations: index 0 = x, index 1 = y.  Thread A(0) stores x then loads y;
// thread B(1) stores y then loads x.  So thread i stores to location i and
// loads from location (1 - i).  Because each thread stores to a location it
// never reads, intra-thread store-to-load forwarding never fires here: every
// load reads shared memory.
//
// SC (store_buffering == false): a store writes straight to shared memory, so the
// reachable states are exactly the interleavings of the four operations.
// TSO (store_buffering == true): a store is placed in the thread's private buffer
// and flushed to shared memory at some later, nondeterministic point; a thread may
// therefore load the other location before its own store (and the other thread's
// store) has been flushed, which is what makes (0,0) reachable.

namespace {

struct State {
    int sh[2];    // shared memory:  sh[0]=x, sh[1]=y
    int pc[2];    // per-thread program counter, 0..2
    int pend[2];  // pend[i]=1: thread i has a buffered store not yet flushed
    int r[2];     // read result, -1 = not read yet (r[0]=r1, r[1]=r2)

    long encode() const {
        long e = 0;
        e = e * 2 + sh[0];
        e = e * 2 + sh[1];
        e = e * 3 + pc[0];
        e = e * 3 + pc[1];
        e = e * 2 + pend[0];
        e = e * 2 + pend[1];
        e = e * 3 + (r[0] + 1);   // -1..1 -> 0..2
        e = e * 3 + (r[1] + 1);
        return e;
    }
};

void explore(const State& s, bool sb, std::set<long>& visited, int& mask) {
    if (!visited.insert(s.encode()).second) return;

    if (s.pc[0] == 2 && s.pc[1] == 2) {           // both reads have completed
        int idx = (s.r[0] << 1) | s.r[1];
        mask |= (1 << idx);
        return;
    }

    // Each thread executes its next program-order operation.
    for (int i = 0; i < 2; i++) {
        if (s.pc[i] >= 2) continue;
        State t = s;
        if (s.pc[i] == 0) {                       // STORE 1 to location i
            if (sb) t.pend[i] = 1;                //   TSO: buffer it
            else    t.sh[i] = 1;                  //   SC : commit immediately
            t.pc[i] = 1;
        } else {                                  // LOAD from location (1 - i)
            t.r[i] = t.sh[1 - i];                 //   read shared memory
            t.pc[i] = 2;
        }
        explore(t, sb, visited, mask);
    }

    // TSO only: flush any thread's buffered store to shared memory.
    if (sb) {
        for (int i = 0; i < 2; i++) {
            if (!s.pend[i]) continue;
            State t = s;
            t.sh[i] = 1;
            t.pend[i] = 0;
            explore(t, sb, visited, mask);
        }
    }
}

// Path-count every SC interleaving (no state dedup: distinct paths = distinct
// interleavings) and tally the outcome of each.
void count_rec(const int sh[2], const int pc[2], const int r[2], int counts[4]) {
    if (pc[0] == 2 && pc[1] == 2) {
        counts[(r[0] << 1) | r[1]]++;
        return;
    }
    for (int i = 0; i < 2; i++) {
        if (pc[i] >= 2) continue;
        int sh2[2] = {sh[0], sh[1]};
        int pc2[2] = {pc[0], pc[1]};
        int r2[2]  = {r[0], r[1]};
        if (pc[i] == 0) { sh2[i] = 1; pc2[i] = 1; }        // store
        else            { r2[i] = sh2[1 - i]; pc2[i] = 2; } // load
        count_rec(sh2, pc2, r2, counts);
    }
}

} // namespace

int allowed_outcomes(bool store_buffering) {
    State s{};
    s.sh[0] = s.sh[1] = 0;
    s.pc[0] = s.pc[1] = 0;
    s.pend[0] = s.pend[1] = 0;
    s.r[0] = s.r[1] = -1;
    std::set<long> visited;
    int mask = 0;
    explore(s, store_buffering, visited, mask);
    return mask;
}

void sc_outcome_histogram(int counts[4]) {
    for (int i = 0; i < 4; i++) counts[i] = 0;
    int sh[2] = {0, 0}, pc[2] = {0, 0}, r[2] = {-1, -1};
    count_rec(sh, pc, r, counts);
}
