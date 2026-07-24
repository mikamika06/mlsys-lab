#pragma once
// The underlying C++ implementation your bridge must call. Defined in
// main.cpp (playing the role of "the rest of the C++ program").
int cpp_add(int a, int b);

// Your .cpp must define a function that a C-style consumer can link
// against by the unmangled name `add` -- see task.md. Deliberately not
// declared here: the whole bug is about what language linkage your own
// definition does or doesn't give it.
