#pragma once

// Called once per value you decide to "process". Marked noinline so the
// real compiler emits a genuine, distinct linker symbol for every DIFFERENT
// T it gets instantiated with -- main.cpp counts these directly with `nm`
// on its own compiled binary; it does not trust your logic, it inspects
// the real object code.
template <typename T>
__attribute__((noinline)) void process(T x) {
    volatile T sink = x;
    (void)sink;
}

// Must "process" five heterogeneous values: an int, a float, a double, and
// two different strings (const char*).
//
// The THREE NUMERIC values (int, float, double) must all be routed through
// exactly ONE shared instantiation: convert each to `double` yourself
// before calling process<double>(...), instead of calling process<int>,
// process<float>, and process<double> separately -- that produces three
// DISTINCT symbols for values that are all "just a number". The two
// strings already share the type `const char*`, so calling
// process<const char*> on each of them is already just ONE instantiation.
//
// Done correctly, main.cpp's `nm` scan finds exactly 2 distinct
// process<T> symbols in the compiled binary (one for double, one for
// const char*) instead of 4 (one each for int, float, double, const char*).
void processAll();
