## Context

A gather, `out[i] = data[idx[i]]`, has no guaranteed spatial locality —
`idx` can jump anywhere in `data` from one `i` to the next. But it often
has *temporal* locality hiding in it: the same index can appear more than
once across the request stream. Processed in the original `i` order,
though, that reuse is invisible if the two occurrences are far apart —
by the time you get back to an index you've already fetched, hundreds of
unrelated lines have gone through the cache since, and the old line is
long gone. The data dependency is fake: nothing requires you to honor
`idx`'s original order while *reading* — only `out[i]` has to end up in
the right place.

**Segmenting** the gather means sorting (or bucketing) the request
stream by target index before you touch memory, so every repeat of the
same index becomes back-to-back instead of scattered, then scattering
the results back to their original `out[i]` slots. The memory traffic
drops without changing a single value in `out`.

## Task

Implement

```cpp
void segmented_gather(const float* data, int dsize, const int* idx, int n, float* out);
```

Compute `out[i] = data[idx[i]]` for every `i` in `[0, n)`. Call
`touch((long)idx[i] * sizeof(float))` exactly once per `i` (`n` touches
total) — but you may process the `n` requests in any order you choose.
Sort (or otherwise group) them by `idx[i]` so repeats of the same index
are touched consecutively, then write each result to its **original**
`out[i]` position — regardless of processing order, `out` must come out
identical to a straight `i`-order fill.

## Example

`idx = [0, 5, 0]`: touching in that order, addresses `0, 20, 0` (bytes)
are touched with `20` in between — the second `0` re-fetches. Sorted by
index first (`[0, 0, 5]`, original positions `[0, 2, 1]`), the two `0`s
land back-to-back — the second one hits — then `5` is touched once.
Either way, `out = [data[0], data[5], data[0]]`.

## What the gate checks

The driver builds 4096 floats of data (4x a modelled 4096-byte cache's
capacity) and 8192 gather requests: the first 4096 sweep every element
once in index order, the second 4096 request the exact same 4096
elements again in the same order. It runs a fixed naive in-order gather
(the harness's own baseline) and your `segmented_gather` on independent
fresh caches, and prints both miss counts plus a checksum of `out`. The
grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$
\mathrm{exact\_match} = 1 \iff \text{naive\_misses, learner\_misses, and checksum all match the reference}
$$

The reference reports `naive_misses=512 learner_misses=256
checksum=8394752.000000` — every element the naive pass revisits has
long since been evicted (its repeats cost exactly as much as its first
visit, `512 = 256 + 256`), while sorting by index groups every repeat
right after its first occurrence, halving the misses to `256` — one
compulsory miss per distinct line, zero repeats. A stub that leaves
`out` untouched fails the checksum immediately, on top of never calling
`touch()` at all.
