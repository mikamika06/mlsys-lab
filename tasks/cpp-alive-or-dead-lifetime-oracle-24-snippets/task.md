## Context

In C++, an object's **lifetime** begins when its initialization is complete and ends when its destructor begins execution (or when its storage is released/reused for trivially destructible types).
Dangling pointers or references are created when one attempts to access an object outside of its lifetime.

Consider the following struct under the LP64 ABI:
```cpp
struct Gadget {
    int id;
    void* buffer;
};
```

## Task

You must determine whether the target `Gadget` object is **ALIVE (True)** or **DEAD (False)** at the exact moment execution reaches `/* MARK */` in each of the 24 snippets below.

1. `void f() { Gadget g; /* MARK (target = g) */ }`
2. `void f() { { Gadget g; } /* MARK (target = g) */ }`
3. `Gadget* p = new Gadget; delete p; /* MARK (target = object pointed to by p) */`
4. `Gadget* p = new Gadget; /* MARK (target = object pointed to by p) */ delete p;`
5. `const Gadget& ref = Gadget(); /* MARK (target = the temporary object) */`
6. `Gadget&& ref = Gadget(); /* MARK (target = the temporary object) */`
7. `Gadget* f() { Gadget g; return &g; } void g() { Gadget* p = f(); /* MARK (target = the object p points to) */ }`
8. `const Gadget& f() { return Gadget(); } void g() { const Gadget& p = f(); /* MARK (target = the temporary object) */ }`
9. `void f() { static Gadget g; /* MARK (target = g) */ }`
10. `void f() { Gadget g; auto l = [&g]() { /* MARK (target = g) */ }; l(); }`
11. `auto f() { Gadget g; return [&g]() { /* MARK (target = g) */ }; } void h() { f()(); }`
12. `void f() { auto p = std::make_unique<Gadget>(); /* MARK (target = the object p points to) */ }`
13. `void f() { auto p = std::make_unique<Gadget>(); p = nullptr; /* MARK (target = the original object) */ }`
14. `void f() { auto p1 = std::make_shared<Gadget>(); { auto p2 = p1; } /* MARK (target = the object p1 points to) */ }`
15. `void f() { auto p1 = std::make_shared<Gadget>(); std::weak_ptr<Gadget> w = p1; p1.reset(); /* MARK (target = the object w points to) */ }`
16. `Gadget* g; void init() { g = new Gadget; } void destroy() { delete g; } int main() { init(); destroy(); /* MARK (target = object g points to) */ }`
17. `void f(const Gadget& g) { /* MARK (target = g) */ }; void h() { f(Gadget()); }`
18. `void f() { Gadget g; std::move(g); /* MARK (target = g) */ }`
19. `void f() { alignas(Gadget) char buf[sizeof(Gadget)]; Gadget* p = new (buf) Gadget; p->~Gadget(); /* MARK (target = object p points to) */ }`
20. `Gadget g; void f() { /* MARK (target = g) */ }`
21. `void f() { std::vector<Gadget> v; v.push_back(Gadget()); /* MARK (target = the temporary Gadget()) */ }`
22. `void f() { std::vector<Gadget> v(1); Gadget* p = &v[0]; v.clear(); /* MARK (target = the object p pointed to) */ }`
23. `void f() { thread_local Gadget g; /* MARK (target = g) */ }`
24. `void f() { for(int i=0; i<1; ++i) { Gadget g; /* MARK (target = g) */ } }`

Implement

```cpp
int predict_lifetimes(bool out[24]);
```

- Write your True/False prediction for snippet `i+1` into `out[i]` (so `out[0]`
  is snippet 1, `out[23]` is snippet 24).
- Return your prediction for `sizeof(Gadget)` in bytes under LP64.

## Example

For a (much shorter) 2-snippet version where snippet 1 is alive and snippet 2
is dead, `out` would end up `{true, false, ...}` and the function would
`return 16;`.

## What the gate checks

`main.cpp` is a fixed driver that actually **runs** all 24 snippets for real,
using an instrumented `Gadget` whose constructors/destructor record their
`this` pointer in a global "currently alive" set. At each snippet's MARK point
the driver asks that set "is this exact address alive right now?" — pure
pointer-identity comparison, so even the dangling-pointer/dangling-reference
cases are inspected without ever reading through invalid memory (no UB). This
gives a real, runtime-computed ground truth for every snippet, never a
hardcoded answer.

The driver then calls your `predict_lifetimes`, compares each of your 24
predictions against that ground truth, and prints a line per snippet plus the
total match count and the size check. The reference gets all 24 snippets and
the size right, so its printed output is the canonical one; the grader
recompiles your `solve.cpp` against the same fixed driver and requires the
printed output to match the reference's **exactly** ($\mathrm{exact\_match}=1.0$).
Guessing "always alive" or "always dead" only matches on the snippets where
that happens to be correct and fails the rest — you have to actually reason
about each lifetime rule (scope exit, `delete`, reference lifetime extension,
`static`/`thread_local`/global storage duration, `unique_ptr`/`shared_ptr`/
`weak_ptr` ownership, container reallocation/`clear()`, placement-new +
explicit destructor calls, and `std::move` being a no-op cast) to get all 24.
