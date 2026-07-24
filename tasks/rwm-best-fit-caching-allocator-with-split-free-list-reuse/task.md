## Context

Frameworks like PyTorch never call `cudaMalloc`/`cudaFree` for every tensor —
device allocation is slow and fragments quickly under a training loop. Instead
they run a **caching allocator** in front of the device: it holds onto freed
memory in a pool and services new requests from that pool whenever possible,
only growing the *reserved* (device-owned) region on a genuine cache miss.

A simplified version of that allocator behaves like this for a request of
$n$ bytes:

1. **Round.** Requests are rounded up to a fixed block granularity
   $B = 512$:
   $$
   \mathrm{round}(n) = B \cdot \left\lceil \frac{\max(n, 1)}{B} \right\rceil .
   $$
   (So any $n \le B$ rounds to exactly $B$.)

2. **Best-fit.** Search the free-block pool for the block with the
   *smallest size* that is still $\ge \mathrm{round}(n)$. If several free
   blocks tie on size, pick whichever of them has been sitting in the pool
   the longest (i.e. scan the pool in the order blocks were inserted and
   keep the first one you see at the minimal qualifying size).

3. **Split.** If a fitting free block is found and it is larger than
   $\mathrm{round}(n)$, carve off exactly $\mathrm{round}(n)$ bytes to serve
   the request and put the remainder back in the pool as a new free block.
   If it matches exactly, no split is needed.

4. **Grow on miss.** If no free block is big enough, grow the allocator's
   *reserved* total by $\mathrm{round}(n)$ bytes and serve the request from
   that freshly reserved memory — unless doing so would exceed the
   allocator's fixed `capacity`, in which case the request fails
   (out-of-memory) and reserved memory does **not** change.

Freeing a block returns it to the end of the free-block pool at its current
(possibly already-split) size; this simplified allocator does not coalesce
adjacent free blocks back together.

## Task

Implement `CachingAllocator` in `solve.py`:

```python
class CachingAllocator:
    def __init__(self, capacity):
        ...

    def malloc(self, nbytes):
        ...

    def free(self, block_id):
        ...
```

* `capacity` — maximum total bytes the allocator may ever reserve from the
  device.
* `malloc(nbytes)` — request `nbytes` bytes following the round / best-fit /
  split / grow-on-miss rules above. Returns an opaque block id (any hashable
  value your implementation chooses) on success, or `None` on OOM.
* `free(block_id)` — return a block previously returned by a successful
  `malloc` to the free-block pool. `free` is never called with an id from a
  failed `malloc`.

Your instance must expose a `reserved` attribute (an `int`) holding the
current total bytes reserved from the device, updated after every call.

## Example

```python
a = CachingAllocator(capacity=2048)

x = a.malloc(100)     # rounds to 512, no free blocks -> reserved becomes 512
y = a.malloc(600)     # rounds to 1024, miss -> reserved becomes 1536
a.free(x)              # the 512-byte block goes back to the pool
z = a.malloc(400)     # rounds to 512, best-fit reuses x's block; reserved stays 1536
w = a.malloc(700)     # rounds to 1024; 1536 + 1024 > 2048 -> OOM, w is None
```

## What the gate checks

The grader implements the same allocator independently as an oracle and
replays several fixed malloc/free traces (referencing earlier mallocs by a
symbolic name so it can free the id *your* allocator actually returned).
After every operation it records `(malloc succeeded, reserved)` and compares
the full sequence, across all traces, to the oracle's. The traces are built
to force every mechanism at least once: an initial miss that grows reserved,
a free followed by an exact best-fit reuse, a free followed by a best-fit
reuse that must split (and the split remainder being fit-reused again), and
a request that must OOM because the free pool has nothing big enough and
`capacity` is already exhausted.
