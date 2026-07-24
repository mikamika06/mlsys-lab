#pragma once
// Classify which of the 12 expressions in task.md are usable in a
// constant-expression context (a core constant expression, i.e. valid as an
// array bound / non-type template argument / static_assert operand).
//
// Return a 12-bit mask: bit i (i = 0..11) must be 1 if and only if
// expression number (i + 1) is a constant expression. Bits 12..31 must be 0.
unsigned classify_constexpr();
