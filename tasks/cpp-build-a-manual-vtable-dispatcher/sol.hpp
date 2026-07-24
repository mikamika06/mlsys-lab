#pragma once
// Base/Derived1/Derived2 use REAL C++ virtual dispatch; their memory layout
// follows this platform's real (Itanium) C++ ABI: a hidden vptr as the
// very first word of the object, pointing at a vtable whose slot 0 holds
// the function pointer for the first virtual member declared in Base
// (`compute`), because it is declared before the destructor.

struct Base {
    virtual double compute(int x) const = 0;
    virtual ~Base() = default;
};

struct Derived1 : Base {
    int factor;
    double weight;
    Derived1(int f, double w) : factor(f), weight(w) {}
    double compute(int x) const override {
        return static_cast<double>(x * factor) + weight;
    }
};

struct Derived2 : Base {
    long offset;
    explicit Derived2(long o) : offset(o) {}
    double compute(int x) const override {
        return static_cast<double>(x) + static_cast<double>(offset);
    }
};

// Call obj->compute(x) WITHOUT using virtual-call syntax (no obj->compute),
// dynamic_cast, or typeid: read the vptr out of `obj`'s raw bytes yourself,
// pull the function pointer out of vtable slot 0, and invoke it directly,
// passing `obj` as the implicit `this` argument.
double manual_dispatch(const Base* obj, int x);
