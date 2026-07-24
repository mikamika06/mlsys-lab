#pragma once
// Runs the SFINAE-overload classification test for 9 scalar types (bool,
// char, short, int, long, long long, float, double, and a pointer) and
// prints one line per type: `<name> <tag>` where `<tag>` is 1, 2, or 3 for
// whichever `process` overload uniquely matched, or `<name> NoMatch` if
// none did.
//
// Implement this entirely in your .cpp: it needs your `process` template
// overloads' definitions in the same translation unit that instantiates
// them, which is true of function templates in general (unlike ordinary
// functions, you can't declare a template here and define it elsewhere).
void run_overload_tests();
