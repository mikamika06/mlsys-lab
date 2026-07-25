## Context

An out-of-order CPU can have many memory requests outstanding at once — that
overlap is **memory-level parallelism (MLP)**, and it's what hides most of
DRAM's latency in practice. But overlap only helps between accesses that
don't depend on each other. Reading `array[0]`, `array[1]`, ..., `array[15]`
directly, all 16 addresses are known before any of them returns, so all 16
can be in flight simultaneously. Chasing a linked list — `p = p->next` — is
the opposite extreme: you cannot even *compute* the address of the next
load until the previous one's result comes back, so no matter how wide the
machine is, the loads happen one at a time.

The relevant number isn't how many memory accesses there are — it's the
length of the longest **dependency chain** among them: the minimum number of
serial round trips required even with unlimited parallelism elsewhere.

## Task

Implement, in `solve.cpp`:

```cpp
int dependency_chain_length(const std::vector<int>& depends_on);
```

`depends_on[i]` is the index of the access that access `i` must wait for
(`-1` if it doesn't wait for anything; whenever it isn't `-1`,
`depends_on[i] < i`). Compute, for each `i`, the length of the chain ending
at `i` (`1` if `depends_on[i] == -1`, else `1 + chain_length_ending_at(depends_on[i])`),
and return the maximum over all `i`.

## Example

The driver (`main.cpp`, fixed) builds four 16-access patterns:

- **pointer_chase** — access `i` depends on access `i-1` for every `i`: one
  unbroken chain of length 16.
- **independent** — every access is `-1`: 16 one-step chains, longest is 1.
- **two_chains** — 2 interleaved pointer chases (`i` depends on `i-2` once
  `i >= 2`): 2 chains of length 8 each.
- **four_chains** — 4 interleaved pointer chases: 4 chains of length 4 each.

```
pointer_chase n=16 chain_length=16
independent n=16 chain_length=1
two_chains n=16 chain_length=8
four_chains n=16 chain_length=4
```

The same 16 memory accesses take 16x longer in the worst case
(`pointer_chase`) than the best (`independent`), purely because of how the
addresses depend on each other — nothing about the *data* itself changed.
Splitting one long chain into `K` independent chains of the same total
length divides the critical path by `K` (`two_chains`, `four_chains`): this
is exactly why software prefetching and manually unrolling a linked-list
walk into several interleaved pointers can speed up pointer-chasing code
without changing what it computes.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Returning a
simple count of non–`-1` entries (instead of following each chain to its
end) gets `pointer_chase` right (all 15 non-root entries) but is wrong on
`two_chains` and `four_chains`, where several *shorter* chains coexist and
the answer is the length of the longest one, not the total count of
dependent accesses.
