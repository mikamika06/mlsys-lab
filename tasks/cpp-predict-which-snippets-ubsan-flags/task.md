## Context

Clang's **UndefinedBehaviorSanitizer** (`-fsanitize=undefined`) instruments
C++ code at compile time to catch Undefined Behavior (UB) at runtime. Common
categories:

1. **Signed integer overflow**: $a+b$, $a-b$, or $a \times b$ exceeding the
   32-bit signed range $[-2^{31}, 2^{31}-1]$. **Unsigned** overflow (modulo
   $2^{32}$) is, by contrast, **well-defined** in C++ and is *never* flagged.
2. **Misaligned pointer dereference**: dereferencing a pointer to type $T$
   at an address $k$ with $k \bmod \mathrm{alignof}(T) \ne 0$.
3. **Shift out of range**: shifting a 32-bit `int` by a negative amount, or
   by $\ge 32$.
4. **Division / modulo by zero**.
5. **Array index out of bounds** (when the compiler can determine the
   array's size, even if the index itself is only known at runtime).

## Task

Implement

```cpp
void predict_ubsan_flags(int out[15]);
```

filling `out[i]` with `1` if UBSan reports snippet `i+1` below, `0` if it
doesn't:

```cpp
//  1. int a=2000000000,b=1500000000; volatile int c=a+b;
//  2. int a=100,b=100; volatile int c=a*b;
//  3. unsigned int a=4000000000u,b=1000000000u; volatile unsigned int c=a+b;
//  4. unsigned int a=3000000000u,b=2u; volatile unsigned int c=a*b;
//  5. alignas(8) unsigned char buf[16]={}; int* p=(int*)(buf+1); volatile int v=*p;
//  6. alignas(8) unsigned char buf[16]={}; int* p=(int*)(buf+4); volatile int v=*p;
//  7. alignas(8) unsigned char buf[24]={}; double* p=(double*)(buf+1); volatile double v=*p;
//  8. int x=1; volatile int y=x<<35;
//  9. int x=1; volatile int y=x<<5;
// 10. int x=1, s=-1; volatile int y=x<<s;
// 11. int a=10,b=0; volatile int c=a/b;
// 12. int a=10,b=0; volatile int c=a%b;
// 13. int a=10,b=5; volatile int c=a/b;
// 14. int arr[5]={1,2,3,4,5}; volatile int idx=7; volatile int v=arr[idx];
// 15. int arr[5]={1,2,3,4,5}; volatile int idx=3; volatile int v=arr[idx];
```

## Example

Snippet `int a=2000000000,b=1500000000; volatile int c=a+b;` overflows the
32-bit signed range ($2000000000+1500000000 > 2^{31}-1$), so UBSan flags it:
prediction `1`. Snippet `int a=100,b=100; volatile int c=a*b;` stays well
within range: prediction `0`.

## What the gate checks

For each of the 15 snippets, the driver writes out its exact source, really
compiles it with `clang++ -fsanitize=undefined -fno-sanitize-recover=all`,
and really runs the resulting binary: with `-fno-sanitize-recover=all`, a
detected UB aborts the process (nonzero exit), while well-defined code exits
cleanly (`0`). That real pass/fail is the ground truth, never hardcoded or
simulated.

The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference across all 15 predictions.
