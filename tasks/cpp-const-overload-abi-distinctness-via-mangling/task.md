## Context

C++ member functions can be overloaded on the `const` qualification of the implicit object parameter: a class may declare both `int read();` and `int read() const;`. On the real Itanium C++ ABI (what clang++ actually emits on macOS), these are two entirely separate functions with two separate linker symbols -- the const overload's mangled name carries a `K` qualifier marker that the non-const overload's does not:

```
_ZN6Widget4readEv     -- Widget::read()
_ZNK6Widget4readEv    -- Widget::read() const   (the extra 'K')
```

`main.cpp` proves both symbols genuinely exist in the compiled binary by shelling out to `nm` on its own executable at runtime -- the real symbol table the real compiler produced, not a hand-computed prediction.

But `const` on a member function is a weaker promise than it looks: it only means "this function won't write to non-`mutable` members". `Widget::calls` here is declared `mutable`, which means the compiler will happily let a `const` member function write to it anyway. A `const` overload that uses that loophole is not really read-only, even though it type-checks.

## Task

`Widget` (declared in `sol.hpp`) has `mutable int calls;` and the two `read()` overloads above. Fix `Widget::read() const` in `solve.cpp` so it behaves as a genuine read-only view: it must return `calls` **without** modifying it, even though `mutable` would legally let it. (`Widget::read()`, the non-const overload, is correct as shipped: it increments `calls` and returns the new value.)

## Example

```cpp
Widget w{0};
w.read();              // mutable: calls 0 -> 1, returns 1
w.read();              // mutable: calls 1 -> 2, returns 2

const Widget& cw = w;
cw.read();              // const: must return 2, calls stays 2
cw.read();              // const: must still return 2
```

## What the gate checks

`main.cpp` calls `read()` twice through a mutable `Widget&` (must show two increasing values, e.g. `1 2`) and twice through a `const Widget&` (must show the *same* value both times, since a correct const read must not mutate `calls`). It also confirms via `nm` on its own executable that `_ZN6Widget4readEv` and `_ZNK6Widget4readEv` both really exist as distinct symbols (they always will here, since the two overloads have different signatures by construction -- that part is a fact about the real ABI, not something your implementation controls). Your printed numbers are compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Using the `mutable` loophole to increment `calls` inside the const overload makes the two const reads print two different, still-increasing numbers instead of the same one twice.
