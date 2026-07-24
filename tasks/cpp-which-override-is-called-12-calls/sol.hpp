#pragma once
// "Which override is called?" — virtual dispatch classification.
//
// A fixed class hierarchy (see task.md) makes 12 member-function calls through
// a mix of objects, base/derived pointers, and references. Each override
// returns a distinct integer tag:
//     Base::who      = 1
//     Derived::who   = 2
//     MoreDerived::who = 3
//     Base::nonvirt    = 10
//     Derived::nonvirt = 20
//
// Fill out[0..11] with the tag of the override that ACTUALLY runs at each of
// the 12 call sites listed (in order) in task.md.
void predict_tags(int out[12]);
