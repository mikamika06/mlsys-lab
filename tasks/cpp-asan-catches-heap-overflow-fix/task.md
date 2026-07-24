## Context

A **heap buffer overflow** happens when code writes past the end of a
dynamically allocated array. AddressSanitizer (ASan) catches this class of
bug by instrumenting every memory access and aborting the moment one lands
outside its allocation. Without a sanitizer, the same bug is silent: it just
quietly corrupts whatever memory happens to sit next in the heap.

Consider:

```cpp
struct DataChunk {
    int header;
    double values[4];
};

void populate_chunks(DataChunk* chunks, int num_chunks) {
    for (int i = 0; i <= num_chunks; i++) {   // off-by-one!
        chunks[i].header = i;
        for (int j = 0; j < 4; j++)
            chunks[i].values[j] = i * 1.5 + j;
    }
}
```

If `chunks` was allocated with `new DataChunk[num_chunks]`, the `i <=
num_chunks` bound writes one element past the end of the array on the last
iteration — a genuine heap overflow.

## Task

Fix `populate_chunks` in `solve.cpp`:

```cpp
void populate_chunks(DataChunk* chunks, int num_chunks);
```

Write exactly the first `num_chunks` elements of `chunks`:

$$\text{chunks}[i].\text{header} = i, \qquad
\text{chunks}[i].\text{values}[j] = i \cdot 1.5 + j \quad (j \in [0,4))$$

for every $i \in [0, \text{num\_chunks})$ — and touch nothing else.

The fixed driver in `main.cpp` allocates each `chunks` buffer with
`num_chunks + 1` elements: the first `num_chunks` are the real buffer, and
the extra element at index `num_chunks` is a hidden **guard chunk**, pre-set
to a `-999` sentinel and never meant to be written by your function. The
driver prints every populated chunk followed by the guard chunk, so an
overflow that reaches the guard shows up directly in the output.

## Example

For `num_chunks = 2`: `chunks[0] = {header: 0, values: [0, 1, 2, 3]}`,
`chunks[1] = {header: 1, values: [1.5, 2.5, 3.5, 4.5]}`, and the guard chunk
at `chunks[2]` must still read `{header: -999, values: [-999, -999, -999,
-999]}` afterward.

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`). The shipped starter has the
`i <= num_chunks` off-by-one bug: every in-bounds chunk still prints
correctly, but the guard chunk gets overwritten with
`{header: num_chunks, values: [num_chunks*1.5, ...]}` instead of staying
`-999`, so its printed line — and therefore the whole byte-exact
comparison — fails.
