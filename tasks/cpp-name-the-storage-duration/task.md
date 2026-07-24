## Context

C++ objects have a *storage duration* that determines how long the memory
they occupy exists. The four categories:

- **automatic** – local variables inside a function body; a fresh object
  every time the enclosing block is entered, gone when it exits.
- **static** – one object shared for the entire program run (whether it's a
  global, or `static` inside a function).
- **thread_local** – like `static`, but each thread gets its *own*
  instance. Declared with the `thread_local` keyword.
- **dynamic** – allocated with `new`; must be `delete`d manually. The
  pointer variable itself is automatic, but we classify what it *points
  to* as dynamic.

```cpp
void demo_variables() {
    int a;                       // automatic
    static int b;                // static
    thread_local int c;          // thread
    int *d = new int;            // dynamic

    double e;                    // automatic
    static double f;             // static
    thread_local double g;       // thread
    double *h = new double;      // dynamic

    char i;                      // automatic
    static char j;               // static
    thread_local char k;         // thread
    char *l = new char;          // dynamic

    short m;                     // automatic
    static short n;              // static
    thread_local short o;        // thread
    short *p = new short;        // dynamic

    long long q;                 // automatic
    static long long r;          // static
    thread_local long long s;    // thread
    long long *t = new long long; // dynamic
}
```

## Task

Implement

```cpp
void name_storage_durations(std::string out[20]);
```

Fill `out[0..20)` with the label (`"automatic"`, `"static"`, `"thread"`, or
`"dynamic"`) for each of the 20 declarations above, in order.

## Example

The first four entries: `out[0]="automatic"; out[1]="static";
out[2]="thread"; out[3]="dynamic";`

## What the gate checks

`main.cpp` builds the same 5 types x 4 categories for real and derives the
correct label for each **from observed runtime behaviour**, never a
hardcoded table:

- **automatic** is proven by forcing two overlapping activations of the
  same local variable via real recursion (both alive at once) and checking
  they land at *different* addresses — a fresh object per activation.
- **static** vs. **thread_local** is proven by calling the same probe
  function from two real `std::thread`s and comparing the address each one
  sees: a plain `static` local is one shared object (same address in both
  threads); a `thread_local` local is per-thread (different address in
  each thread). The language guarantees both outcomes, so this is a real
  test, not a heuristic.
- **dynamic** is proven with a global `operator new`/`operator delete`
  override that records every address the allocator actually hands out; a
  `new T` pointer is dynamic iff its address is in that set.

The grader compiles your `.cpp` with the real local `clang++`, runs it
against the fixed driver, and requires all 20 printed match lines to equal
the reference's ($\mathrm{exact\_match}=1.0$).
