## Context

A Python generator function is a compact way to describe a state machine. Each
time execution reaches a `yield`, Python suspends the frame and stores enough
information to continue later. Conceptually, a generator has an internal state
$q$ and a transition function that moves it forward:

$$
(q_{t+1}, y_t) = \delta(q_t),
$$

where $q_t$ is the current state and $y_t$ is the next value produced.

The same behavior can be implemented explicitly by storing the state in an
object. The `__next__` method performs one transition and updates the integer
state field. When the machine has no more values, it raises `StopIteration`.

For a countdown sequence, the state is the next integer to emit. Starting from
state $n$, each transition emits the current state and decreases it by one until
the terminal state is reached.

## Task

Implement `countdown(n)`.

The function must return an iterator object that behaves like the generator:

```python
def countdown(n):
    ...
```

For a non-negative integer $n$, calling `next()` repeatedly on the returned
object must produce:

$$
n-1, n-2, \dots, 1, 0
$$

After the last value, `next()` must raise `StopIteration`.

Implement the iterator as an explicit state machine. The returned object must
store its current position in an integer attribute named `state`, and it must
implement `__next__`.

Do not use the `yield` keyword anywhere in the implementation.

## Example

```python
it = countdown(4)

next(it)  # 3
next(it)  # 2
next(it)  # 1
next(it)  # 0
next(it)  # raises StopIteration
```

## What the gate checks

The gate computes a reference sequence by running an independent state-machine
oracle and compares the produced values for several inputs.

It also checks the implementation source code and rejects solutions containing
the `yield` keyword. The returned iterator must expose an integer `state`
attribute and implement `__next__`.
