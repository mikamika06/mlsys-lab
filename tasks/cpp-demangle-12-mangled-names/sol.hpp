#pragma once
#include <string>

// Demangles ONE Itanium-mangled C++ symbol name, for the restricted subset
// exercised by main.cpp: free functions and (possibly nested, possibly
// const) member functions, taking `void`/`int`/`double` parameters, each
// optionally wrapped in one level of pointer (P) or reference (R), and P/R
// optionally further wrapped in const (K). All names start with "_Z".
//
// Grammar for this subset:
//   _Z <name-or-nested> <params>
//   <name-or-nested> := <len><chars>                     -- e.g. "3foo"
//                      | N [K] <len><chars>+ E            -- e.g. "N1S1fE", "NK1S1fE"
//   <params> := "v"                                       -- no arguments
//             | <type>+                                   -- one or more, run together
//   <type> := "i"                 -- int
//           | "d"                 -- double
//           | "P" ["K"] ("i"|"d") -- pointer, optionally to const
//           | "R" ["K"] ("i"|"d") -- reference, optionally to const
//
// Output format must match the real Itanium demangler exactly:
//   - nested names are joined with "::"
//   - no return type is ever printed
//   - parameters are comma-and-space separated inside "(...)"
//   - `const` on a pointer/reference target is written AFTER the base type,
//     e.g. "double const*", "double const&" (not "const double*")
//   - a const member function (the "K" right after "N") gets " const"
//     appended at the very end, after the closing ")"
//
// Examples:
//   "_Z3fooii"              -> "foo(int, int)"
//   "_ZN1S1fEv"              -> "S::f()"
//   "_Z4funcPKd"             -> "func(double const*)"
//   "_ZNK5Outer1fEPi"        -> "Outer::f(int*) const"
std::string demangleOne(const std::string& mangled);
