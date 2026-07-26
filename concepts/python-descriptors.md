---
title: "What is python descriptors?"
description: "Python descriptors explained, with a measured protocol-call count for data vs non-data precedence you can reproduce, plus graded exercises."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is python descriptors?

A descriptor in python is any object whose class defines `__get__`, `__set__`, or `__delete__`,
so that attribute lookup calls those methods instead of reading `__dict__` directly. Which method
fires depends entirely on one precedence rule: a descriptor that defines `__set__` wins over an
instance's own `__dict__` entry every time, while one that defines only `__get__` loses to that
same entry, 0 calls out of 5 reads. Below, both counts are measured directly by instrumenting
the protocol methods themselves.

## How it works

Every `obj.attr` lookup that finds `attr` on the type runs through
`type.__getattribute__`'s fixed search order: first check whether `type(obj)` defines `attr` as
a **data descriptor** (it has `__set__` or `__delete__`) — if so, call its `__get__`
unconditionally, before ever looking at `obj.__dict__`. Otherwise check `obj.__dict__` itself.
Only if that fails does a **non-data descriptor** (`__get__` alone) or a plain class attribute
get a turn. `__slots__`, covered in [python slots](python-slots.md), is the most common
non-data-descriptor sighting most Python programmers never notice as one: each slot is a
*member descriptor* with both `__get__` and `__set__`, which is precisely why a slotted
attribute cannot be shadowed by an instance dict — slotted classes don't have one.

This precedence rule is not academic; the standard library leans on it as a feature.
`functools.cached_property` is deliberately built as a **non-data** descriptor: on the first
access its `__get__` computes the value and writes it straight into `obj.__dict__` under the
same public name. Every access after that finds the instance-dict entry before lookup ever
reaches the descriptor again, so `__get__` runs exactly once per instance no matter how many
times the attribute is read — the same "0 calls after the dict is populated" effect the table
below measures directly, here used on purpose instead of being a bug.
[Building it from scratch](../tasks/pyt-build-cached-property-from-scratch/task.md) and
[fixing a cached property that never caches](../tasks/pyt-fix-a-cached-property-that-never-caches/task.md)
both gate on getting that exact call count right.

`__set_name__` runs at a different time entirely: once per descriptor attribute, while
`type.__new__` is still assembling the class body, before any instance exists. It is how a
descriptor created as `width = NamedField()` — with no name passed to its constructor — learns
that it was bound to `width`, per PEP 487.
[Wiring that hook through a metaclass](../tasks/pyt-set-name-descriptor-naming-via-metaclass-path/task.md)
is one of the tasks below.

Descriptors are one of several places this bank measures "the order things are checked changes
the answer": [memory coalescing](memory-coalescing.md) asks which 128-byte segment an address
lands in before asking what value it holds, and [cache blocking](cache-blocking.md) shows that
the *order* a loop nest visits the same addresses can change a miss count by two orders of
magnitude without changing a single computed result. Attribute lookup is the same idea with a
fixed, four-step search order instead of a hardware cache, and the order is exactly what makes
it predictable enough to count.

## Protocol calls measured against lookup precedence

The script below defines a data descriptor (`Data`, with `__get__` and `__set__`) and a
non-data descriptor (`NonData`, `__get__` only), reads each 5 times, writes the *same name*
directly into the instance `__dict__` — bypassing the descriptor entirely, the way a raw
`__dict__` poke or a plain `obj.attr = x` assignment on a `__set__`-less descriptor both do —
then reads and writes again and counts every protocol-method call.

| lookup path | protocol method that fires | calls |
|---|---|---|
| `h.data` (data), before instance-dict shadow, 5 reads | `Data.__get__` | 5 |
| `h.data` (data), after instance-dict shadow, 5 reads | `Data.__get__` (shadow ignored — data wins) | 5 |
| `h.nondata` (non-data), before instance-dict shadow, 5 reads | `NonData.__get__` | 5 |
| `h.nondata` (non-data), after instance-dict shadow, 5 reads | *none — instance dict wins* | 0 |
| `h2.data = v`, 4 writes | `Data.__set__` | 4 |
| `h2.nondata = v`, 4 writes | *none — plain dict write, no `__set__` exists* | 0 |
| class body executes (`Holder` defines `data`, `nondata`) | `__set_name__`, once each | 2 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
SETNAME_LOG = []

class Data:                                  # data descriptor: __get__ AND __set__
    def __init__(self):
        self.get_calls = 0
        self.set_calls = 0
    def __set_name__(self, owner, name):
        self.name = name
        SETNAME_LOG.append((owner.__name__, name))
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self.get_calls += 1
        return obj.__dict__.get("_" + self.name, 0)
    def __set__(self, obj, value):
        self.set_calls += 1
        obj.__dict__["_" + self.name] = value

class NonData:                               # non-data descriptor: __get__ only
    def __init__(self):
        self.get_calls = 0
    def __set_name__(self, owner, name):
        self.name = name
        SETNAME_LOG.append((owner.__name__, name))
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self.get_calls += 1
        return "via-descriptor"

class Holder:
    data = Data()
    nondata = NonData()

print(f"set_name calls at class creation: {len(SETNAME_LOG)} -> {SETNAME_LOG}")

h = Holder()
READS = 5
for _ in range(READS): h.data
for _ in range(READS): h.nondata
print(f"before shadow: data.__get__={Holder.__dict__['data'].get_calls} "
      f"nondata.__get__={Holder.__dict__['nondata'].get_calls}")

h.__dict__["data"] = "SHADOW"                # poke the instance dict directly
h.__dict__["nondata"] = "SHADOW"
d0, n0 = Holder.__dict__["data"].get_calls, Holder.__dict__["nondata"].get_calls
for _ in range(READS): h.data
for _ in range(READS): h.nondata
d1, n1 = Holder.__dict__["data"].get_calls, Holder.__dict__["nondata"].get_calls
print(f"after shadow: data.__get__ +{d1-d0} more, nondata.__get__ +{n1-n0} more")

WRITES = 4
h2 = Holder()
for v in range(WRITES): h2.data = v
print(f"{WRITES} writes to h2.data -> Data.__set__ fired {Holder.__dict__['data'].set_calls} times")
g0 = Holder.__dict__["nondata"].get_calls
for v in range(WRITES): h2.nondata = v
print(f"{WRITES} writes to h2.nondata -> NonData.__get__ +{Holder.__dict__['nondata'].get_calls - g0} "
      f"(nondata in h2.__dict__: {'nondata' in h2.__dict__})")
PY
```

Read the table top to bottom as one story: before the shadow, both descriptors behave
identically — 5 calls for 5 reads, because with nothing in the instance `__dict__` yet there is
nothing to compete with either one. The moment `data` and `nondata` both get an entry planted
under the same name, they diverge completely. The data descriptor's `__get__` keeps firing 5
more times, oblivious to the shadow, because step one of attribute lookup never even checks
`__dict__` when a data descriptor exists. The non-data descriptor's `__get__` fires *zero* more
times — the instance dict entry is now permanently in front of it in the search order, forever,
for that instance. `__set_name__` is the outlier row: it belongs to a different moment
entirely, firing exactly twice, once per descriptor, while the class body is being built, and
never again for any instance created afterward.

## Practise it

```bash
mlsys grade pyt-fix-a-non-data-descriptor-shadowed-by-the-instance-dict
```

[That task](../tasks/pyt-fix-a-non-data-descriptor-shadowed-by-the-instance-dict/task.md) gates
`exact_match == 1.0` against a grader that constructs several `Widget` instances, reassigns
`.level` after construction, and — the check this task is really about — pokes
`w.__dict__["level"]` directly and requires that `w.level` still reads back clamped. The shipped
`Clamped` descriptor implements only `__get__`, exactly the `NonData` row above: it is a
non-data descriptor, so a plain `w.level = 500` assignment writes straight into the instance
dict and the clamp is gone for good after the first write. Adding a `__set__` is the entire fix
— it is the one line that moves `Clamped` from the bottom row of the table to the top one.

In roughly increasing difficulty:
[classify data vs non-data descriptors](../tasks/pyt-data-vs-non-data-descriptor-classifier/task.md) (no code, from class metadata alone),
[wire `__set_name__` through a metaclass](../tasks/pyt-set-name-descriptor-naming-via-metaclass-path/task.md),
[implement `property` from scratch](../tasks/pyt-implement-property-from-scratch/task.md),
[reimplement `__getattribute__`'s resolution order](../tasks/pyt-reimplement-getattribute-resolution-order/task.md),
and [hand-implement slot descriptors](../tasks/pyt-hand-implement-slot-descriptors/task.md) without
using `__slots__` at all.

## Common mistakes

- **Assuming any `__get__` makes a data descriptor.** Precedence is decided by `__set__` or
  `__delete__` alone. A class with `__get__` only loses to the instance dict — 0 calls in the
  table above — no matter how carefully its `__get__` is written.
- **Thinking a read-only `property` behaves like a non-data descriptor.** `property` always
  defines `__set__` at the type level, even when no setter function was given — it just raises
  `AttributeError` inside that `__set__` instead of assigning. A read-only property is still a
  data descriptor and still cannot be shadowed, which is the opposite of what its "read-only"
  name suggests.
- **Expecting `functools.cached_property` to work on a `__slots__` class.** It is a non-data
  descriptor that caches by writing to `obj.__dict__`; a slotted instance has no `__dict__` to
  write to, so the very first access raises `TypeError` instead of caching anything — the
  precedence trick it depends on requires the thing [python slots](python-slots.md) removes.
- **Believing the shadow only happens through a manual `__dict__` poke.** An ordinary
  `obj.attr = value` on a `__set__`-less descriptor is doing the identical thing: with no
  `__set__` to intercept it, plain assignment writes to the instance dict by default, and that
  is the whole bug in the task above.
- **Forgetting `__set_name__` only runs once, at class-creation time.** Code that expects it to
  fire again when an instance is created, or when the attribute is reassigned, will find it
  simply never called — the table's 2 is a permanent count, not a per-instance one.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[Python Morsels](https://www.pythonmorsels.com/exercises/paths/)** — Trey Hunner's paid
  weekly-exercise subscription has a named descriptors learning path with an explanatory
  article on `__get__`/`__set__`/`__set_name__` and data-vs-non-data precedence, followed by
  small locally-run exercises. The single closest topical match found for this exact subject;
  $14-29/month, and it does not count protocol calls.
- **[Advanced Python Mastery (David Beazley)](https://github.com/dabeaz-course/python-mastery)**
  — free, and its "Inside Python Objects" section covers descriptors and slots almost
  one-to-one with this page's mechanism, with full worked solutions to check against by hand.
  No automated grading.
- **[Fluent Python, 2nd ed.](https://github.com/fluentpython/example-code-2e)** — Luciano
  Ramalho's book has a full chapter built around the descriptor protocol, including the
  data-vs-non-data distinction, with runnable example code per chapter. Reading only, nothing
  to submit.
- **[Exercism's Python track](https://exercism.org/tracks/python)** — real auto-grading
  exists, and a "Descriptors" concept exercise is one of its ~146, but it is one exercise
  inside a much larger general-Python track rather than dedicated coverage of precedence.
- **[wtfpython](https://github.com/satwikkansal/wtfpython)** — good guess-then-reveal coverage
  of the surrounding data-model gotchas (mutable defaults, identity, interning), but it does
  not walk through the descriptor protocol itself.

## References

1. Python documentation, *Data model* — "Implementing Descriptors" and "Invoking Descriptors".
   https://docs.python.org/3/reference/datamodel.html#implementing-descriptors
2. Hettinger, R., *Descriptor HOWTO* — the canonical worked walkthrough, including the pure-Python
   `__getattribute__` equivalent this page's table is built from.
   https://docs.python.org/3/howto/descriptor.html
3. PEP 487, *Simpler customisation of class creation* — the `__set_name__` hook.
   https://peps.python.org/pep-0487/
