## Context

A *Treiber stack* is the classic lock-free stack. It holds a single atomic
pointer `head` to a singly-linked list of nodes and mutates it only through
compare-and-swap (CAS):

$$\textbf{push}(v):\quad n.next \leftarrow head;\quad \text{CAS}(head,\ n.next,\ n)\ \text{(retry on failure)}$$

$$\textbf{pop}():\quad t \leftarrow head;\ \text{if } t=\varnothing\ \text{return empty};\quad \text{CAS}(head,\ t,\ t.next)\ \text{(retry on failure)}$$

Because every mutation is a single atomic CAS, multiple producers and consumers
can operate at once with no lock. The correctness property that must hold for
*every possible interleaving* is **multiset conservation**: the multiset of
values returned by `pop` equals the multiset of values handed to `push` -- no
element is ever lost, duplicated, or corrupted, and none is invented.

The subtle failure modes this catches: a non-atomic (read head, then write head)
update lets two threads splice over each other, dropping or duplicating nodes; a
CAS that forgets to retry silently discards a push; reading `head->next` without
a proper CAS loses items under contention.

## Task

Implement `TreiberStack::push(int)` and `TreiberStack::pop(int&)` in `solve.cpp`
using an atomic CAS loop on the provided `std::atomic<Node*> head_`.

- `push(value)` allocates a node and links it onto the top of the stack.
- `pop(out)` removes the top node; it returns `true` and writes the value into
  `out` when the stack is non-empty, or returns `false` when the stack is empty.

Both must be safe and lock-free under concurrent callers. You may leak popped
nodes -- safe reclamation is a separate problem and is not graded here.

The fixed driver in `main.cpp` starts 4 producer threads (which together push
the 100000 distinct values `0 .. 99999`) and 4 consumer threads that pop
concurrently until the stack is drained and all producers have finished. It
gathers every popped value, sorts them, and prints aggregate checks.

## Example

With a correct implementation the sorted popped multiset is exactly
`{0, 1, ..., 99999}` no matter how the threads interleave, so the driver prints:

```
count=100000
sum=4999950000
xor=<fixed value>
conserved=1
```

A stub that drops pushes or never pops yields an empty result:

```
count=0
sum=0
xor=0
conserved=0
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires an **exact match** of the
printed numbers against the reference (`count`, `sum`, `xor`, `conserved`).
Because a correct lock-free stack conserves the multiset, those numbers are
deterministic across runs even though the thread schedule is not; any lost or
duplicated element changes `count`, `sum`, `xor`, or `conserved` and fails the
gate.
