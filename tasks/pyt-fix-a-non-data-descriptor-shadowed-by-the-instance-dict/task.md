## Context

Python's attribute lookup (`obj.attr`) checks, in order:

1. Does `type(obj)` (or a base class) define `attr` as a **data descriptor**
   — an object with both `__get__` *and* (`__set__` or `__delete__`)? If so,
   its `__get__` wins, unconditionally.
2. Otherwise, does `obj.__dict__` contain `attr`? If so, that value wins.
3. Otherwise, does `type(obj)` define `attr` as a **non-data descriptor**
   (only `__get__`) or a plain class attribute? Use that.

The subtlety: a **non-data** descriptor sits at step 3, *below* the
instance's own `__dict__`. The moment something writes `attr` directly into
`obj.__dict__` — including an ordinary `obj.attr = value` assignment, since
with no `__set__` to intercept it, that's exactly what a plain assignment
does — the descriptor is permanently shadowed for that instance. Every
future read of `obj.attr` finds the instance-dict entry first and never
calls the descriptor's `__get__` again.

Adding a `__set__` method (even a trivial one) turns the descriptor into a
**data** descriptor, promoting it to step 1 — now it wins the lookup no
matter what ends up in `obj.__dict__` under that name.

## Task

`starter.py` defines a `Clamped` descriptor meant to keep a public attribute
pinned to $[\mathrm{lo}, \mathrm{hi}]$:

$$
\mathrm{clamp}(v) = \max\bigl(\mathrm{lo},\ \min(\mathrm{hi},\ v)\bigr)
$$

and a `Widget` class that uses it for its `level` attribute (default range
$[0, 100]$). The clamp is only enforced the *first* time, in `__init__` —
after that, plain assignment silently stores the raw, unclamped value
straight into the instance's `__dict__`, because `Clamped` only implements
`__get__`.

Fix `Clamped` by adding a `__set__(self, obj, value)` that stores
`clamp(value)` in the instance under `self.private_name` (already computed
in `__init__` as `"_" + name`) — do not write to the public name. This
makes `Clamped` a data descriptor, so it enforces the clamp on *every*
assignment and can never be shadowed by the instance's own `__dict__`.

Do not change `__init__`, `__get__`, `private_name`, or `Widget`.

## Example

```python
w = Widget(50)
w.level = 500
print(w.level)          # -> 100  (clamped, not 500)

# The instance dict is never shadowed under the public name:
w.__dict__["level"] = 12345   # simulate something poking it directly
print(w.level)                # -> still 100 — the data descriptor wins
```

## What the gate checks

The grader builds several `Widget` instances and checks, all in one
pass:

* construction with in-range and out-of-range values clamps correctly, and
  the instance `__dict__` never picks up a `"level"` key (only the private
  one);
* two instances don't share state;
* reassigning `.level` after construction still clamps;
* the regression case this task is really about: directly writing
  `w.__dict__["level"] = 12345"` (bypassing normal attribute assignment
  entirely) must **not** change what `w.level` reads back — a real data
  descriptor always wins the lookup, no matter what's sitting in the
  instance's own `__dict__` under that name.

**exact_match** is `1.0` only if every one of these checks passes; any
single failure (or an exception) makes it `0.0`.
