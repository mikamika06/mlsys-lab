## Context

Ring attention distributes a long sequence across $p$ devices. Each device owns a local query block and a local key/value block. During computation, key/value blocks are rotated around a ring so every device can attend to every context partition.

Assume each device stores a key/value block containing $s$ tokens. Each token has hidden width $d$ and both key and value tensors use $b$ bytes per element. The bytes in one key/value block are

$$
B_{\mathrm{KV}} = 2 \cdot s \cdot d \cdot b ,
$$

where the factor of $2$ accounts for the key and value tensors.

For $p$ devices, a ring schedule performs $p-1$ rotations. Each rotation sends one key/value block from every device to its neighbor. The communication volume per device is therefore

$$
C_{\mathrm{device}} = (p-1) B_{\mathrm{KV}},
$$

and the total bytes moved across all links are

$$
C_{\mathrm{total}} = p(p-1)B_{\mathrm{KV}} .
$$

The task models communication volume, not execution time or bandwidth.

## Task

Implement `ring_attention_comm`:

```python
def ring_attention_comm(num_devices: int, seq_per_device: int,
                        hidden_dim: int, bytes_per_element: int) -> tuple[int, int]:
    ...
```

Return a tuple:

1. `bytes_per_step`: the KV bytes transmitted by one device during one ring rotation.
2. `total_bytes`: the total bytes transferred across all devices for the complete ring schedule.

Use integer arithmetic only.

The inputs satisfy:

- $num\_devices \ge 2$
- $seq\_per\_device > 0$
- $hidden\_dim > 0$
- $bytes\_per\_element > 0$

## Example

```python
bytes_per_step, total = ring_attention_comm(4, 1024, 4096, 2)

# bytes_per_step = 16777216
# total = 201326592
```

The example uses a KV block of

$$
2 \cdot 1024 \cdot 4096 \cdot 2 = 16777216
$$

bytes. There are $3$ rotations and $4$ devices, so

$$
4 \cdot 3 \cdot 16777216 = 201326592 .
$$

## What the gate checks

The gate builds an independent oracle from the ring communication model and compares the returned values exactly for several parameter combinations.

The `modeled_mem_access` score is $1.0$ only when the implementation returns the exact pair

$$
(B_{\mathrm{KV}}, p(p-1)B_{\mathrm{KV}})
$$

computed by the oracle. No approximations or bandwidth assumptions are used.
