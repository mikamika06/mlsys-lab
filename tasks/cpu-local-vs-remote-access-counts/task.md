## Context

On a NUMA (Non-Uniform Memory Access) machine, physical memory is split
into fixed-size *pages*, and each page is pinned to one *NUMA node* (a
CPU socket with its own directly-attached memory controller). A thread
running on a given node can read a page pinned to that same node quickly
(a **local** access); reading a page pinned to a *different* node has to
travel over the inter-socket interconnect, which costs noticeably more
latency (a **remote** access).

Given a fixed page size $\mathrm{page\_bytes}$, the page a byte address
$a$ falls in is

$$
\mathrm{page}(a) = \left\lfloor \frac{a}{\mathrm{page\_bytes}} \right\rfloor,
$$

and every page has one owning node, `node_of_page[page]`. A thread
pinned to `home_node` classifies each of its accesses as local or remote
purely by comparing that page's node against its own.

## Task

Implement

```cpp
void count_local_remote(const long* addrs, int n, long page_bytes,
                         const int* node_of_page, int num_pages,
                         int home_node, long* out);
```

For each of the `n` addresses in `addrs`, compute its page
($\mathrm{page}(a)$ above) and look up that page's node in
`node_of_page`. If the node equals `home_node`, count it as local
(`out[0]`); otherwise count it as remote (`out[1]`). Every address
satisfies `0 <= addrs[i] < num_pages * page_bytes`, so every page lookup
is in range. `out[0] + out[1]` must equal `n`.

## Example

`page_bytes = 4096`, `node_of_page = {0, 1}`, `home_node = 0`. An access
to byte address `100` falls on page `0` (node `0`, local); an access to
byte address `5000` falls on page `1` (node `1`, remote).

## What the gate checks

`exact_match` on the printed `(local, remote)` pair for a fixed 40-access
trace over an 8-page, 2-node placement. Swapping the local/remote
comparison, mixing up which output slot gets which count, or letting
`out[0] + out[1] != n` (e.g. an off-by-one page computation that indexes
`node_of_page` out of its intended range) all change the printed numbers.
