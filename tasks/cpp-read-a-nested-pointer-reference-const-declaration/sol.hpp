#pragma once
// Runs the declaration-classification test for the 12 real C++ types in
// task.md (int*, const int*, int* const, ..., int*&) and prints one line
// per type: its index and its classification label.
//
// Implement this entirely in your .cpp: it needs your classify_type<T>()
// template's definition in the same translation unit that instantiates it
// (true of function templates in general -- you can't declare it here and
// define it elsewhere).
void run_declaration_tests();
