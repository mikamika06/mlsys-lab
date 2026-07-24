## Context

On a NUMA (Non-Uniform Memory Access) machine, memory is physically split
across nodes, and each core can reach its own node's memory faster than a
remote one. Most operating systems place a virtual page's physical memory
lazily, under a policy called **first-touch**: the page has no physical
home until some thread actually reads or writes it, and at that moment the
page is placed on whichever node that thread is running on. Every later
access from a *different* node still works, but it crosses the interconnect
and pays a remote-access penalty — the page never moves again.

This means the *order* of first accesses, not the final access pattern,
decides where every page lives. Two threads that both use a page heavily
still get very different performance if one of them only ever touches it
after the other already claimed it.

## Task

Implement, in `solve.cpp`:

```cpp
void first_touch_owner(const int* thread_of_access, const int* page_of_access, int n,
                        const int* node_of_thread, int num_threads,
                        int num_pages, int* owner_of_page);
```

You are given a trace of `n` memory accesses in chronological order
(`thread_of_access[i]`, `page_of_access[i]`), and a map from thread id to the
NUMA node it runs on (`node_of_thread[t]`, `t` in `[0, num_threads)`). Fill
`owner_of_page[p]` for every page `p` in `[0, num_pages)` with the NUMA node
of whichever thread's access to page `p` occurs **earliest** in the trace —
that is the node first-touch would have placed it on. A page that no access
in the trace ever touches has no owner: report it as `-1`, not `0`.

## Example

The driver (`main.cpp`, fixed) generates a 40-access trace from a seeded
generator, over 4 threads split across 2 NUMA nodes
(`node_of_thread = {0, 0, 1, 1}`) and 10 pages, where page 9 is never
touched by the trace. The first several accesses are:

```
access 0 thread=1 page=3
access 1 thread=3 page=0
access 2 thread=3 page=7
access 3 thread=3 page=6
access 4 thread=1 page=1
...
```

Thread 1 runs on node 0 and thread 3 runs on node 1, so page 3's first touch
(access 0, thread 1) places it on node 0, while page 0's first touch
(access 1, thread 3) places it on node 1 — even though later accesses may
come from either node. The final ownership:

```
page 0 owner=1
page 1 owner=0
page 2 owner=0
page 3 owner=0
page 4 owner=0
page 5 owner=0
page 6 owner=1
page 7 owner=1
page 8 owner=1
page 9 owner=-1
```

The starter reports every page as `-1`, which is wrong for every page except
the untouched one.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output (trace echo plus per-page
ownership) to match the reference (`main.cpp` + `ref.cpp`) byte-for-byte
(`exact_match == 1.0`). Using the node of a page's *last* access instead of
its *first* gets several pages right by luck but disagrees with the
reference wherever a page is touched by both nodes, and forgetting the `-1`
case for an untouched page fails on page 9 alone.
