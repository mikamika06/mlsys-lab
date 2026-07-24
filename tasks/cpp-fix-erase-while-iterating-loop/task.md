## Context

A classic C++ bug: erasing elements from a sequence container (like
`std::vector`) while iterating over it, without accounting for the fact
that erasure shifts every following element one slot left:

```cpp
for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].value < 0) {
        nodes.erase(nodes.begin() + i);   // shifts everything after i left
    }
}
```

After `erase(nodes.begin() + i)`, the element that used to be at index
`i + 1` now sits at index `i`. The loop's `++i` runs unconditionally,
though, so that element is never examined — it's silently kept even if
its own `value` is negative. This bug is invisible when negative values
are isolated, and only shows up when two or more negative values are
**adjacent**.

```cpp
struct DataNode {
    short id;
    int   value;
    long  next;
};
```

## Task

`solve.cpp` ships the buggy version above. Fix `filter_nodes`, in
`solve.cpp`:

```cpp
void filter_nodes(std::vector<DataNode>& nodes);
```

Remove every `DataNode` whose `value < 0` from `nodes`, in place, without
skipping any. After an erase, do **not** advance past the current index —
re-examine whatever now sits there, since it might also need removing.
Only advance when nothing was erased at the current position.

## Example

For nodes with values `[10, -5, -10, 20, -1]` (ids `1..5`), the two
middle values `-5` and `-10` are adjacent. The buggy loop erases `-5`
(id 2), which shifts `-10` (id 3) into slot 1, then advances past it
unconditionally — `-10` survives. A correct implementation removes both,
leaving only `[10, 20]` (ids `1, 4`).

## What the gate checks

The fixed driver (`main.cpp`) runs `filter_nodes` over exactly this
5-node sequence (with the two adjacent negatives as the trap) and prints
`sizeof(DataNode)`, the remaining count, and every remaining node's
`id`/`value`/`next`. The gate is an exact string match
(`exact_match == 1.0`) against the reference's printed output — leaving
even one node behind that should have been removed changes the count and
fails the gate.
