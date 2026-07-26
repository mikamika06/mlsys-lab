## Context

A GPU executes many parallel thread blocks. The number of warps that can be resident on a Streaming Multiprocessor (SM) is limited by several resources:

* **Registers per SM** – each thread consumes a fixed amount of registers.
* **Shared memory per SM** – each block uses a fixed amount of shared memory; the total usage must fit within the available shared memory.
* **Maximum threads per SM** – hardware imposes an upper bound on the number of active threads (often 2048).

Occupancy is the fraction of this maximum that can actually be resident. The resource that yields the smallest possible occupancy *limits* the kernel.

Typical values for a recent NVIDIA GPU are:

| Resource | Typical limit |
|----------|---------------|
| Registers per SM | 65536 |
| Shared memory per SM | 49152 bytes (48 KiB) |
| Maximum threads per SM | 2048 |
| Warp size | 32 |

For a given kernel we assume each thread uses the same number of registers and that every block requests the same amount of shared memory. The function you will implement determines which resource is the bottleneck.

## Task

Implement `identify_limiter`:

```python
def identify_limiter(block_dim: int,
                     regs_per_thread: int,
                     shared_bytes_per_block: int) -> str:
    ...
```

* `block_dim`: number of threads per block.
* `regs_per_thread`: registers consumed by each thread.
* `shared_bytes_per_block`: bytes of shared memory requested by each block.

Return one of the strings:

* `'register'` – register usage limits occupancy,
* `'shared'`   – shared‑memory usage limits occupancy, or
* `'thread'`   – maximum‑threads per SM limit is the bottleneck.

The function should use the typical limits listed above. Do not hardcode the result for specific inputs; compute it from the formulas below.

**Formulas**

Let  

```
max_regs_per_sm      = 65536
max_shared_bytes     = 49152
max_threads_per_sm   = 2048
warp_size            = 32
```

Compute the maximum number of active threads per SM allowed by each resource:

1. **Registers**

```
threads_by_regs = min(max_regs_per_sm // regs_per_thread,
                      max_threads_per_sm)
```

2. **Shared memory**  
   Each block can be resident on an SM only if its shared usage fits.
   The number of blocks that can coexist is  

```
blocks_by_shared = max(1, max_shared_bytes // shared_bytes_per_block)
```

Hence the maximum threads due to shared memory are

```
threads_by_shared = min(blocks_by_shared * block_dim,
                        max_threads_per_sm)
```

3. **Thread count limit**

```
threads_by_thread_cap = max_threads_per_sm
```

The limiting resource is the one that yields the smallest value among  
`threads_by_regs`, `threads_by_shared`, and `threads_by_thread_cap`.  Return its label.

## Example

```python
>>> identify_limiter(128, 32, 0)   # 128 threads per block, no shared memory
'register'

# The register limit is 65536 // 32 = 2048, equal to the SM thread cap,
# but because regs_per_thread == 32, it matches the thread cap exactly.
# Since registers and thread cap tie, the function prefers 'register'.

>>> identify_limiter(128, 32, 49152)   # each block uses all shared memory
'shared'

# Here blocks_by_shared = 1 → threads_by_shared = min(1*128,2048)=128,
# which is far less than the register limit of 2048 (and the thread cap
# of 2048), so 'shared' is returned. With shared_bytes_per_block equal
# to the entire SM shared memory, each block consumes all shared memory
# and only one block can run, giving 128 threads total — the smallest
# of the three limits.
```

## What the gate checks

The grader runs a deterministic reference implementation that follows the formulas above and compares its output string exactly (`exact_match`). If your function returns the same string for all inputs it passes; otherwise the gate fails.
