## Context

Ring attention spreads a long sequence over $p$ devices. Device $r$ permanently
owns query block $r$ and key/value block $r$. On each step every device hands
its current key/value block to its right-hand neighbour and receives one from
the left, so after $p-1$ steps every block has visited every device.

A block holds $s$ tokens of width $d$; keys and values are separate tensors of
$b$ bytes per element. Counting bytes on the wire is the whole model here — no
bandwidth, no latency, no overlap with compute.

Under a **causal** mask the schedule is cheaper than it first looks. Query block
$r$ only attends to key/value blocks $0 \dots r$; anything later is fully
masked, so those arrivals compute nothing. A block therefore does not need to
finish the loop — it only needs to reach the last device that still has unmasked
work for it, and the hop after that carries bytes nobody reads.

Dropping those hops does two things. It changes the cluster total by a constant
factor, and — less obvious, and the reason this task asks for it — it stops
spreading the traffic evenly. In the dense schedule every device puts the same
number of bytes on the wire. Under the causal mask they do not, and the device
that forwards the most is not the device that computes the most.

## Task

Implement `ring_attention_comm`:

```python
def ring_attention_comm(num_devices: int, seq_per_device: int,
                        hidden_dim: int, bytes_per_element: int
                        ) -> tuple[int, int, int, tuple[int, ...]]:
    ...
```

Return four things:

1. `bytes_per_step` — the key/value bytes one device puts on the wire during one
   rotation.
2. `total_dense` — bytes moved by the whole cluster over the full schedule when
   every block completes the ring, i.e. no mask.
3. `total_causal` — bytes moved when every block stops as soon as no device
   downstream of it has unmasked work left.
4. `per_device_causal` — a tuple of length `num_devices`. Element $i$ is the
   number of bytes device $i$ *sends* over the whole causal schedule. Forwarding
   a block someone else originated counts as sending it. The elements must sum
   to `total_causal`.

Use integer arithmetic only; every result is exact. Inputs satisfy $p \ge 2$,
$s > 0$, $d > 0$, $b > 0$.

## Example

```python
ring_attention_comm(4, 1024, 4096, 2)
```

```text
(16777216, 201326592, 100663296, (16777216, 33554432, 50331648, 0))
```

One block is $2 \cdot 1024 \cdot 4096 \cdot 2 = 16777216$ bytes. In the dense
schedule four devices each send it across three rotations. Under the causal mask
block $0$ still has to travel to every later device, block $3$ travels nowhere,
and the traffic piles up towards the far end of the line — the last device sends
nothing at all, while its neighbour sends three blocks' worth.

## What the gate checks

`modeled_mem_access` is $1.0$ only when all four results match an independently
derived oracle, for every tested $(p, s, d, b)$ — including $p = 2$, an odd
device count, and a case where $s$ is not a power of two. Both the tuple length
and every element are compared exactly, and non-integers fail.

Returning `total_causal == total_dense` fails. So does a `per_device_causal`
that is flat: the sum can be right while the distribution is wrong, and the
distribution is what the last two tasks in this module build on.
