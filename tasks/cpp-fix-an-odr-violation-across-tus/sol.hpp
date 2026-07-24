#pragma once

// main.cpp defines its own `struct Config { int x; };` and an
// `inline __attribute__((noinline)) int get_size()` returning sizeof(that
// Config) -- with ordinary EXTERNAL linkage (no anonymous namespace), just
// like a real inline function pulled in from a shared header. `noinline`
// forces the compiler to emit it as a genuine callable symbol rather than
// inlining the call away, so this is a real weak/"linkonce" linker symbol,
// not just source text.
//
// You must define your OWN, DIFFERENT `struct Config { int x; double y; };`
// and your OWN `get_size()` returning sizeof(YOUR Config) -- but wrapped in
// an anonymous namespace, so it gets INTERNAL linkage and cannot collide
// with main.cpp's same-named external symbol.
//
// reportSize() must call your get_size() and return whatever it returns.
//
// If your get_size() has ordinary external linkage instead (no anonymous
// namespace, but still `inline` so the link doesn't just fail outright),
// the real linker will silently MERGE it with main.cpp's same-named weak
// symbol. Since main.cpp is always compiled first on the command line, its
// definition wins the merge for the WHOLE program -- including calls made
// from inside your own file -- and reportSize() ends up reporting
// main.cpp's Config size instead of yours.
int reportSize();
