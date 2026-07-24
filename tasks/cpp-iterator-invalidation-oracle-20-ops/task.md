## Context

Container mutation operations (`push_back`, `insert`, `erase`, `reserve`)
can invalidate existing iterators pointing into the container. Getting this
wrong is a classic source of dangling-iterator undefined behavior. The
rules differ by container:

1. **`std::vector<T>`**:
   - `push_back` / `insert`: if $\text{size}+1 > \text{capacity}$,
     reallocation happens and **every** iterator is invalidated. Otherwise,
     iterators at or after the insertion position `op_pos` are invalidated;
     iterators strictly before it remain valid.
   - `erase`: iterators at or after `op_pos` are invalidated; iterators
     strictly before it remain valid.
   - `reserve(N)`: if $N > \text{capacity}$, reallocation happens and every
     iterator is invalidated; otherwise nothing moves and every iterator
     stays valid.

2. **`std::deque<T>`**: any mutation invalidates **every** iterator,
   unconditionally — deque gives none of vector's "iterators before the op
   stay valid" guarantee.

3. **`std::list<T>` and `std::map<Key,Value>`**: `insert` / `push_back`
   never invalidates existing iterators (nodes don't move). `erase`
   invalidates only the iterator that pointed *directly* at the erased
   element (`iter_pos == op_pos`); every other iterator stays valid.

## Task

Implement `classify_iterator_validity` in `solve.cpp`:

```cpp
int classify_iterator_validity(const IterScenario& s);
```

`IterScenario` (declared in `sol.hpp`) bundles `container`, `operation`,
`size`, `capacity`, `iter_pos`, and `op_pos` (for `OP_RESERVE`, `op_pos`
instead carries the requested new capacity `N`). Apply the rules above and
return `1` (still valid) or `0` (invalidated).

The fixed driver in `main.cpp` runs your function over 20 fixed scenarios
spanning vector (reallocating and non-reallocating push_back/insert/erase,
plus reserve), deque, list, and map, and prints the 20 labels.

## Example

```cpp
// vector, insert, size=5, capacity=10, iter_pos=3, op_pos=2
// -> no reallocation (6 <= 10); iter_pos(3) >= op_pos(2) -> invalidated: 0

// list, erase, size=5, iter_pos=1, op_pos=3
// -> different element erased (1 != 3) -> still valid: 1
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across all 20 scenarios. The
starter labels everything "still valid" (`1`), which is wrong for every
scenario that should actually invalidate the iterator.
