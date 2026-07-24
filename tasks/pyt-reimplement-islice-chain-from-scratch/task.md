## Context

The iterator protocol is CPython's uniform interface for "give me the next
value, or tell me you're done": an iterator is any object whose `__next__`
raises `StopIteration` when exhausted, and `iter(x)` obtains one from any
iterable. Generator functions (`def f(): ... yield v`) are the easiest way
to implement the protocol by hand — each `yield` suspends the frame,
handing control (and one value) back to the caller, and resumes exactly
where it left off on the next `next()` call.

`itertools.islice(iterable, start, stop, step)` and
`itertools.chain(*iterables)` are two of the simplest members of that
protocol:

$$
\text{islice}(a, \text{start}, \text{stop}, \text{step}) =
\bigl(a_i \mid i \in \{\text{start}, \text{start}+\text{step}, \dots\},\; i < \text{stop}\bigr)
$$

$$
\text{chain}(a^{(1)}, \dots, a^{(k)}) =
a^{(1)}_0, a^{(1)}_1, \dots, a^{(1)}_{n_1-1},\; a^{(2)}_0, \dots, a^{(k)}_{n_k-1}
$$

Both are lazy: they pull from their source(s) one element at a time and
never materialize the whole input up front, which is what lets
`itertools.islice(huge_or_infinite_generator, 0, 5)` work without reading
past the 5th element.

## Task

Implement two generator functions, without importing `itertools`:

```python
def my_islice(iterable, start: int, stop: int, step: int = 1):
    ...

def my_chain(*iterables):
    ...
```

- `my_islice` must reproduce `itertools.islice(iterable, start, stop,
  step)`: consume `iterable` one element at a time via `iter`/`next`
  (never index into it, never convert it to a `list`/`tuple` first),
  skipping the first `start` elements, then yielding every `step`-th
  element up to (not including) position `stop`, stopping early if the
  source is exhausted first.
- `my_chain` must reproduce `itertools.chain(*iterables)`: yield every
  element of the first iterable, then every element of the second, and so
  on, again consuming each source one element at a time.
- Both must be genuine generators (or otherwise hand-rolled iterators) —
  no delegating to `itertools`, and no eagerly materializing an input
  iterable into a list/tuple before slicing it.

## Example

```python
list(my_islice(range(10), 2, 8, 2))   # [2, 4, 6]
list(my_chain([1, 2], (3,), range(4, 6)))   # [1, 2, 3, 4, 5]
```

## What the gate checks

For several test iterables the grader compares your output, element for
element, against the real `itertools.islice` / `itertools.chain` (a real,
independent oracle) — `islice_exact_match` / `chain_exact_match` must be
`1.0`.

Separately, the grader drains your returned generator with
`collections.deque(gen, maxlen=0)` under a `sys.settrace` line-event
tracer (the same technique `arena.probe.count_line_events` uses) and
compares the number of executed Python-level line events to a floor
derived from how many source elements a real element-by-element
implementation has to step through. `itertools.islice`/`itertools.chain`
themselves, and any eager `list(iterable)[...]` shortcut, run entirely in
C and register at most one Python-level line event no matter how much data
flows through them — so `islice_event_ratio` / `chain_event_ratio` (event
count divided by that floor) must be `>= 1.0`, which only a genuine
element-by-element Python implementation can reach.
