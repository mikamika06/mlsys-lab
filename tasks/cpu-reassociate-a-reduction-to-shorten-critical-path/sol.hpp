#pragma once

// A double value tagged with its dependency-chain DEPTH: the length of
// the longest sequential (data-dependent) chain of additions that
// produced it. A freshly-loaded input value has depth 0. Adding two
// Tracked values produces one whose depth is 1 + max(a.depth, b.depth)
// -- exactly how an out-of-order CPU is limited by the LONGEST
// dependency chain, not the total number of additions (independent
// chains can execute in parallel and overlap in time).
//
// operator+ is a harness-side helper: declared here, defined in
// main.cpp, never in solve.cpp -- it is what makes the depth bookkeeping
// real and unbypassable, not something you compute yourself.
struct Tracked {
    double value;
    int depth;
};

Tracked operator+(Tracked a, Tracked b);

// Reduce x[0..n) into a single sum by REASSOCIATING the reduction into a
// balanced binary tree: split the range into two halves, reduce each
// half independently (recursively), then combine the two partial sums
// with one final operator+. A left-associated serial sum has a critical
// path of n-1; this balanced tree has a critical path of ceil(log2(n)).
// The returned `.value` must equal the exact sum of x[0..n).value; the
// returned `.depth` should fall out of actually building the tree, not
// be set directly.
Tracked reduce_balanced_tree(const Tracked* x, int n);
