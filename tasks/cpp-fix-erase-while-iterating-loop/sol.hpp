#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// PROVIDED (do not change).
// ---------------------------------------------------------------------------
struct DataNode {
    short id;
    int   value;
    long  next;   // opaque "next" handle, not a real pointer
};

// ---------------------------------------------------------------------------
// LEARNER FIXES a bug.
//
// Remove every DataNode whose `value < 0`, IN PLACE, from `nodes`. The
// shipped implementation has the classic erase-while-iterating bug: after
// std::vector::erase() shifts every following element one slot to the
// left, it unconditionally advances its loop index/iterator anyway,
// silently skipping the element that just slid into the erased slot.
// This is invisible on isolated negative values but drops every OTHER
// element whenever two or more negative values are adjacent.
//
// Fix the loop so it only advances when nothing was erased -- after an
// erase, re-examine the element now sitting at the same position, since
// it might also need removing.
// ---------------------------------------------------------------------------
void filter_nodes(std::vector<DataNode>& nodes);
