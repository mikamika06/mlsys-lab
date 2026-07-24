## Context

A Python generator is a state machine. Each suspension point stores a current
state $s_t$, and operations such as `send`, `throw`, and `close` cause transitions
to later states. A driver can observe the values yielded by the generator and
the final termination behavior.

The three important protocol operations are:

- `next(g)` resumes a generator without a value.
- `g.send(x)` resumes it and makes the suspended `yield` expression evaluate to
  $x$.
- `g.throw(E)` resumes it by raising exception $E$ at the suspended `yield`.
- `g.close()` injects `GeneratorExit` and requires the generator to finish
  without yielding another value.

The task models a small protocol where the generator must handle normal input,
a thrown exception, and cleanup.

## Task

Implement `run_protocol()`.

The function must create and drive an internal generator. It must return a list
of observed events. Each event is a tuple containing an operation name and the
observable result.

The scripted driver must perform these operations in order:

1. Start the generator with `next`.
2. Send the string `"alpha"`.
3. Throw `ValueError("bad")`.
4. Send the string `"beta"`.
5. Close the generator.

The internal generator behavior must be:

- The first `next()` yields `"ready"`.
- Receiving a sent value yields `"received:<value>"`.
- Receiving `ValueError("bad")` yields `"handled:bad"`.
- After handling the exception, the next sent value yields `"received:<value>"`.
- Closing the generator runs cleanup and returns `"closed"`.

The returned event list must include yielded values and the final close result.
A close result is recorded from the generator's `StopIteration.value`.

## Example

```python
events = run_protocol()

# events contains:
[
    ("next", "ready"),
    ("send", "received:alpha"),
    ("throw", "handled:bad"),
    ("send", "received:beta"),
    ("close", "closed"),
]
```

## What the gate checks

The gate builds the same scripted driver against a real CPython generator
implementation and compares the returned event sequence. The reference is
computed by executing the generator protocol, not by comparing against manually
written expected output.

A solution must correctly implement the generator state transitions. A function
that only returns the example list without exercising `send`, `throw`, and
`close` will not pass because the gate evaluates the generator behavior through
the protocol implementation.
