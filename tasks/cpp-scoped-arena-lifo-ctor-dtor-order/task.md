## Context

In C++, objects with automatic storage duration in a scope are destroyed in
**strict reverse order of construction** (LIFO: Last-In, First-Out). A *scoped
arena allocator* reproduces that rule manually: it carves object storage out of
one raw buffer with placement-`new`, and on teardown it calls each destructor
explicitly, in the reverse of construction order.

You are given a `Probe` type that records its own lifetime in a global log:

- constructing a `Probe` with id $N$ appends `C<N> `,
- destroying it appends `D<N> `.

`Probe` is non-copyable and non-movable, so it can only be created with
placement-`new` and destroyed with an explicit destructor call — exactly what a
real arena does.

Under the LP64 C++ ABI (`char`=1, `int`=4, `double`=8, natural alignment), the
`Probe` layout `{char tag; int id; double weight;}` occupies

$$S = \operatorname{sizeof}(\text{Probe}) = 16 \text{ bytes}$$

(1 byte `tag`, 3 bytes padding, 4 bytes `id`, 8 bytes `weight`), and an arena
holding all $n$ probes needs

$$M_{\text{arena}} = n \times S.$$

## Task

Implement `long run_scoped_arena(const int* ids, int n)` in `solve.cpp`:

1. reserve a single raw byte buffer large enough for `n` `Probe`s,
2. construct one `Probe` per id in `ids[0..n)` in **forward** order
   (placement-`new`),
3. tear the arena down, destroying every `Probe` in strict **LIFO**
   (reverse-construction) order,
4. return the arena footprint in bytes, `n * sizeof(Probe)`.

All construction and destruction must finish before the function returns, so the
global event log holds the complete `C.../D...` sequence afterward. The `Probe`
type, the log, and the driver are fixed — only `run_scoped_arena` is yours.

## Example

For `ids = {7, 3, 9, 2, 5, 8}` a correct arena emits:

```
C7 C3 C9 C2 C5 C8 D8 D5 D2 D9 D3 D7
arena_bytes=96 object_size=16 count=6
```

Note the destruction ids `8 5 2 9 3 7` are the construction ids reversed.
Destroying in forward order, or using a `std::vector<Probe>` (which destroys
front-to-back), would print the wrong `D...` sequence.

## What the gate checks

The driver compiles (`clang++ -O2 -std=c++20`), runs, and compares the full
printed output — the exact ctor/dtor event sequence **and** the arena byte
accounting — against the reference. The gate is exact string match
($\mathrm{exact\_match} = 1.0$); any deviation in ordering or size fails.
