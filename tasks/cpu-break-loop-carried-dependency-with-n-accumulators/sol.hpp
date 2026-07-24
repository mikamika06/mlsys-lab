#pragma once
// A double value tagged with its dependency-chain DEPTH: the length of the
// longest sequential (data-dependent) chain of additions that produced it.
// A freshly-loaded input value has depth 0. Adding two Tracked values
// produces one whose depth is 1 + max(a.depth, b.depth) -- exactly how an
// out-of-order CPU is limited by the LONGEST dependency chain, not the
// total number of additions (independent chains can execute in parallel
// and overlap in time).
//
// operator+ is a harness-side helper: declared here, defined in main.cpp,
// never in solve.cpp -- it is what makes the depth bookkeeping real and
// unbypassable, not something you compute yourself.
struct Tracked {
    double value;
    int depth;
};

Tracked operator+(Tracked a, Tracked b);

// Reduce x[0..n) into a single sum using `num_accumulators` INDEPENDENT
// accumulators (so their addition chains have no dependency on each
// other) to shorten the critical path, then combine the partial sums.
// The returned `.value` must equal the exact sum of x[0..n).value; the
// returned `.depth` should be as small as possible -- see task.md for how
// it's graded.
Tracked reduce_with_accumulators(const Tracked* x, int n, int num_accumulators);
