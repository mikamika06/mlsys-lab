#pragma once

// Instrumented probe, DEFINED in main.cpp: counts how many times its
// constructor and destructor actually run.
extern int g_ctor_count;
extern int g_dtor_count;

struct Probe {
    int a;
    double b;
    Probe(int a_, double b_);
    ~Probe();
};

// Construct exactly ONE Probe(a, b) into the raw storage `buf` using
// PLACEMENT NEW (`buf` is at least sizeof(Probe) bytes and already aligned
// to alignof(Probe) -- guaranteed by the driver), copy its two fields into
// *out_a / *out_b, then end its lifetime with an EXPLICIT destructor call
// `p->~Probe()`. Never `delete` or `::operator delete` here: `buf` was not
// allocated by `::operator new`, so there is no heap block to free -- only
// the object's lifetime needs ending.
void placement_lifecycle(void* buf, int a, double b, int* out_a, double* out_b);
