#pragma once
#include <string>
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Classify whether the compiler may assume two pointers of the given types
// do NOT alias (i.e. the strict-aliasing rule lets it treat the two accesses
// as independent and reorder/optimize freely).
//
//   Return 1 -> the compiler MAY assume no-alias.
//   Return 0 -> the pointers MAY alias (compiler must be conservative).
//
// `type_a` / `type_b` are type spellings such as "int", "unsigned int",
// "const float", "char", "std::byte", or a class name like "Derived".
//
// `hierarchy` is a list of (derived, base) pairs describing a single-
// inheritance class hierarchy; a class with no base does not appear as a
// `derived` entry with a non-empty base (it simply has no ancestor).
//
// Rules (C++ strict aliasing, simplified):
//   1. Strip leading "const " / "volatile " / "unsigned " / "signed "
//      qualifiers from both type spellings before comparing.
//   2. If either stripped type is "char" or "std::byte", the pointers MAY
//      alias (char/byte are allowed to alias anything) -> return 0.
//   3. If the stripped types are identical, they MAY alias -> return 0.
//   4. If one stripped type is a base class of the other anywhere in the
//      hierarchy chain (in either direction), they MAY alias -> return 0.
//   5. Otherwise the compiler MAY assume no-alias -> return 1.
// ---------------------------------------------------------------------------
int may_assume_no_alias(const std::string& type_a, const std::string& type_b,
                         const std::vector<std::pair<std::string, std::string>>& hierarchy);
