## Context

A real Python program does not provide a simple guarantee that multiple CPU-bound
threads execute simultaneously. The Global Interpreter Lock (GIL) controls which
thread may execute Python bytecode at a given moment. This task uses a simplified
deterministic model rather than measuring real thread timing.

A modeled scheduler receives several instruction streams. Each thread owns a list
of instruction counts between possible checkpoints. The scheduler starts with
thread $0$ and gives it execution until its accumulated instruction count reaches
the switch interval $s$. At that point, the next runnable thread in round-robin
order receives the lock. A thread that finishes its stream is skipped.

For threads $T_0, T_1, \dots, T_{n-1}$, the schedule is the sequence of thread
IDs that hold the GIL at each checkpoint. The model advances through total
executed instructions:

$$
I_{next} = I_{current} + c_i
$$

where $c_i$ is the next checkpoint distance for the current thread. A switch
occurs whenever the accumulated count reaches or exceeds the interval $s$.

## Task

Implement `gil_schedule(interval, streams)`:

```python
def gil_schedule(interval: int, streams: list[list[int]]) -> list[int]:
    ...
```

Arguments:

- `interval` is the modeled GIL switch interval and is a positive integer.
- `streams` is a list where `streams[i]` contains positive instruction counts
  for thread $i$.
- Each value in a stream represents the amount of work until the next checkpoint.

Return a list of thread IDs. Each element represents the thread holding the GIL
when that checkpoint is reached.

Use the deterministic round-robin model:

1. Begin with thread $0$.
2. Consume checkpoints from the current thread in order.
3. Append the current thread ID for every consumed checkpoint.
4. After a checkpoint causes the accumulated instructions for the current owner
   to reach or exceed `interval`, select the next unfinished thread by increasing
   thread ID and wrapping around.
5. Stop after all instruction streams are consumed.

## Example

```python
streams = [[3, 4, 5], [2, 8], [6]]
result = gil_schedule(7, streams)

# [0, 0, 1, 2, 1, 0]
```

The first thread reaches the switch interval after its second checkpoint, so the
scheduler moves to thread $1$. The sequence continues by selecting unfinished
threads in round-robin order.

## What the gate checks

The gate computes the expected schedule using the same deterministic scheduling
model implemented independently inside the checker and compares the submitted
function output exactly. Cases include uneven thread lengths, interval boundaries,
and streams where some threads finish early.

The check is a modeled simulation. It does not depend on operating system thread
timing or the machine's actual GIL behavior.
