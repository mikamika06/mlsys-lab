## Context

On a NUMA machine, data is already placed across nodes, but which node's
cores a *thread* runs on is a separate, schedulable choice. If thread
$t$'s accesses to already-placed data are distributed as
$\mathrm{access\_count}[t][n]$ (how many of its accesses target data on
node $n$), then pinning it to node $n$ makes exactly
$\mathrm{access\_count}[t][n]$ of its accesses local and the rest -- to
data on every other node -- remote:

$$
\mathrm{remote}(t, n) = \Big(\sum_{n'} \mathrm{access\_count}[t][n']\Big) - \mathrm{access\_count}[t][n]
$$

With $T$ threads and $N$ nodes each with room for exactly `capacity`
threads ($T = N \cdot \mathrm{capacity}$), the scheduler must choose an
assignment of threads to nodes -- respecting that capacity -- that
minimizes total remote accesses summed over every thread.

This is a genuine combinatorial optimization, not a per-thread pick: if
several threads all want the same crowded node, a greedy rule like
"assign each thread to its own favorite node, first-come-first-served"
can lock in a bad trade. The right move when a node is full is to bump
whichever contending thread loses the **least** by settling for its
next-best node -- not simply whichever thread happened to be considered
last.

## Task

Implement:

```cpp
long min_remote_accesses(int T, int N, int capacity, const long* access_count);
```

`access_count[t * N + n]` is thread `t`'s access count to node `n`.
Search over every assignment of the `T` threads to the `N` nodes that
gives each node exactly `capacity` threads, and return the minimum
total remote-access count over all such assignments (a small
backtracking search over the `T` threads, one node choice at a time,
respecting each node's remaining capacity, is sufficient at the sizes
this task uses).

## Example

Threads 0, 1, 2 all prefer node 0 most (`access_count` of `100`, `90`,
and `80` there respectively), but node 0 has room for only 2. Thread 0's
next-best node gives it only `10` (a loss of `90` if displaced); thread
2's next-best gives `5` (a loss of `75`); thread 1's next-best gives
`85` (a loss of only `5`). The optimal assignment keeps threads 0 and 2
on node 0 and moves thread 1 to its next-best node -- displacing
whichever thread loses the least, not whichever thread was considered
first.

## What the gate checks

`main.cpp` runs two fixed 6-thread, 3-node, capacity-2 scenarios: one
where three threads contend for one crowded node (case 1) and one with
no contention at all, where every node's two favorites already fit
(case 2), and prints the minimum remote-access total for each. The
candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`)
against the reference's. A first-come-first-served greedy (assign each
thread to its own best node with remaining capacity, never reconsider)
matches the reference on the uncontended case 2 (`60`) but returns `220`
instead of the true optimum `150` on the contended case 1, because it
displaces whichever thread runs out of room rather than whichever one
loses the least.
