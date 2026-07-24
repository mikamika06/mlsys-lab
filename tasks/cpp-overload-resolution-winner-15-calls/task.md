## Context

C++ overload resolution picks a single function from a candidate set by (1)
discarding non-viable candidates, then (2) ranking the remaining ones by how
"good" the conversion is for each argument (Exact Match > Promotion >
Conversion > ... ), and (3) applying tie-breaking rules — e.g. a
non-template function beats an equally-good template specialization, and an
rvalue-reference overload beats a const-lvalue-reference overload for an
rvalue argument.

Some of these rules are famously counter-intuitive:

- **Array-to-pointer decay is ranked as an *Exact Match***, the same rank as
  binding a reference directly to the array — so a plain `f(int*)` overload
  can beat a `template<class T> f(T&)` overload for an array argument, once
  the non-template tie-break applies.
- A **forwarding-reference template** (`template<class T> f(T&&)`) binds to
  a non-const lvalue with *no* added qualification, while
  `f(const std::string&)` requires adding `const` — so for a non-const
  lvalue the template can "steal" the call away from a seemingly more
  specific overload. This flips back once the argument is itself `const`.

## Task

Below are 15 independent scenarios. Each declares exactly two overloads,
tagged to return `0` or `1`, and one call expression. Implement

```cpp
void predict_overload_winners(int out[15]);
```

filling `out[i]` with which tag (`0` or `1`) wins overload resolution for
scenario `i+1`.

```cpp
// 1.  int pick(int);              double pick(double);        pick(5)
// 2.  short pick(short);          int pick(int);               pick(5)
// 3.  int& pick(int&);            const int& pick(const int&); int x=7; pick(x)
// 4.  const int& pick(const int&); int&& pick(int&&);          pick(5)
// 5.  template<class T> pick(T);  int pick(int);                pick(5)
// 6.  struct Base{}; struct Derived:Base{};
//     int pick(Base&); int pick(Derived&);                     Derived d; pick(d)
// 7.  int pick(void*);            int pick(bool);               pick(nullptr)
// 8.  int pick(int);              int pick(...);                pick(5)
// 9.  struct Wrapper{ operator int() const; };
//     int pick(int); int pick(double);                          Wrapper w; pick(w)
// 10. struct S{ int pick() const; int pick(); };                S obj; obj.pick()
// 11. same S as #10                                             const S obj{}; obj.pick()
// 12. int pick(int*);             template<class T> pick(T&);   int arr[5]; pick(arr)
// 13. template<class T> pick(T&&); int pick(const std::string&); std::string s="hi"; pick(s)
// 14. same overloads as #13                                     const std::string s="hi"; pick(s)
// 15. int pick(char*);            int pick(const char*);        char buf[4]="hi"; char* p=buf; pick(p)
```

For scenarios with two same-named overloads, tag `0` is always the first
one listed, tag `1` the second.

## Example

For `int pick(int){return 0;} double pick(double){return 1;} pick(5)`
(scenario 1's shape): `5` is an `int` literal, an exact match for
`pick(int)`, so the winner is tag `0`.

## What the gate checks

The driver defines all 15 scenarios as real, separately-namespaced C++ code
and simply calls each one, using whatever tag the real compiler's overload
resolution actually returns as ground truth — never hardcoded, never
simulated. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference across all 15 predictions.
