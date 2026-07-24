#pragma once
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Classify each of these 15 fixed C++ expressions (with `int x = 1;` in
// scope wherever `x` appears) by its value category -- "lvalue",
// "xvalue", or "prvalue" -- IN THIS ORDER:
//
//    0. x
//    1. 42
//    2. std::move(x)
//    3. "hello"
//    4. std::string("tmp")
//    5. *(&x)
//    6. x + 1
//    7. ++x
//    8. x++
//    9. static_cast<int&&>(x)
//   10. std::declval<int&>()
//   11. std::declval<int&&>()
//   12. std::string("a") + std::string("b")
//   13. std::move(*(&x))
//   14. (x = 1)
//
// Rule (C++17): an expression is a glvalue if its evaluation determines
// the identity of an object; otherwise it is a prvalue. Among glvalues,
// an xvalue denotes an object about to be moved from; the remaining
// glvalues are lvalues.
//
// A real, mechanical way to determine this for any given expression E:
// `decltype((E))` (note the DOUBLE parentheses, which is significant)
// yields `T&` if E is an lvalue, `T&&` if E is an xvalue, and plain `T`
// (no reference) if E is a prvalue -- so std::is_lvalue_reference /
// std::is_rvalue_reference on that type tells you the category directly.
// ---------------------------------------------------------------------------
std::vector<std::string> classify_value_categories();
