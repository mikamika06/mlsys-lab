---
title: "What is python slots?"
description: "Python slots explained, with a measured bytes-per-instance table across attribute counts you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is python slots?

Python slots are a class-level declaration, `__slots__`, that replaces an instance's private
dictionary with a fixed-size array of attribute descriptors. On CPython 3.11, a three-attribute
instance costs 352 bytes with a dict and 56 without one — 6.29 times less — but that ratio is
not a constant, and the table below shows exactly where it stops holding.

## How it works

A normal class gives every instance a pointer to a heap-allocated dictionary. Attribute access
goes through that dictionary: `obj.x` is a hash lookup, `obj.new_attr = 1` is an insert, and the
dictionary can grow to hold anything at any time. `__slots__` changes the class layout instead
of the instance: the class computes one fixed storage cell per declared name and gives each
cell a *member descriptor* — an object whose `__get__`/`__set__` index directly into that cell.
There is no hash table, no growth, and normally no per-instance dictionary allocation at all.

That fixed layout is the same trade CPU and GPU code make when they replace a flexible,
indirect structure with a flat array: [false sharing](false-sharing.md) is what a flat CPU
array can still get wrong at the cache-line level, and [memory coalescing](memory-coalescing.md)
is the GPU version of the same "fixed slot beats scattered lookup" argument, just measured in
128-byte transactions instead of bytes.

Slots take two things away, and both are easy to forget. First, instances can no longer receive
attributes outside the declared set — assigning an undeclared name on a slotted instance raises
`AttributeError: 'Slotted' object has no attribute 'd'`, an error that never fires on a
dict-backed instance. Second, slotted instances cannot be weakly referenced by default:
`weakref.ref(obj)` raises `TypeError: cannot create weak reference to 'Slotted' object` unless
`"__weakref__"` is itself listed among the slots, which is exactly what
[the weak-reference slot classifier task](../tasks/pyt-weakref-slot-requirement-classifier/task.md)
grades you on predicting.

There is a third, more common way to lose the benefit entirely: subclassing. If a subclass of a
slotted class does not itself declare `__slots__`, CPython silently gives that subclass
instances a `__dict__` again, because "no slots" is still the default. Every instance of the
subclass pays full dict-plus-slot-array cost, and nothing in the source of the subclass looks
wrong — [fix a slots-defeating subclass](../tasks/pyt-fix-a-slots-defeating-subclass/task.md) is
built around exactly this bug. A related restriction: only one class in an inheritance chain
may define slots that actually carry instance state, so mixing two non-empty slotted bases
raises `TypeError` at class-creation time rather than at instance-creation time.

None of this is about speed. Attribute access through a slot descriptor and through a
dictionary are both O(1) on average; the difference `__slots__` buys is memory layout, which is
what the measurement below isolates.

## Bytes per instance measured against attribute count

The table varies only the number of declared instance attributes and measures the real total
size of one instance each way: `getsizeof(obj) + getsizeof(obj.__dict__)` for the dict-backed
class, `getsizeof(obj)` for the slotted one — because a dict-backed instance's true cost is the
object header plus the separate dictionary object it points to, and reporting only the header
(56 bytes, constant) would hide most of the cost.

| instance attributes | dict-backed bytes | slotted bytes | ratio |
|---|---|---|---|
| 1 | 352 | 40 | 8.80 |
| 3 | 352 | 56 | 6.29 |
| 6 | 352 | 80 | 4.40 |
| 10 | 352 | 112 | 3.14 |
| 20 | 352 | 192 | 1.83 |
| 30 | **1,640** | 272 | 6.03 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import sys, tracemalloc

def make_classes(n):
    names = [f"a{i}" for i in range(n)]
    body = "\n".join(f"        self.{nm} = {i}" for i, nm in enumerate(names))
    ns = {}
    exec(f"class Dict_{n}:\n    def __init__(self):\n{body}\n", ns)
    exec(f"class Slots_{n}:\n    __slots__ = ({', '.join(map(repr, names))},)\n"
         f"    def __init__(self):\n{body}\n", ns)
    return ns[f"Dict_{n}"], ns[f"Slots_{n}"]

for n in (1, 3, 6, 10, 20, 30):
    D, S = make_classes(n)
    d, s = D(), S()
    dict_total = sys.getsizeof(d) + sys.getsizeof(d.__dict__)
    slot_total = sys.getsizeof(s)
    print(f"n={n:<3} dict={dict_total:<5} slots={slot_total:<4} ratio={dict_total/slot_total:.2f}")

# bulk cross-check: marginal bytes per instance across many instances of one class
D3, S3 = make_classes(3)
N = 50_000
tracemalloc.start()
b0 = tracemalloc.get_traced_memory()[0]
bulk_d = [D3() for _ in range(N)]
b1 = tracemalloc.get_traced_memory()[0]
tracemalloc.stop(); del bulk_d
tracemalloc.start()
b2 = tracemalloc.get_traced_memory()[0]
bulk_s = [S3() for _ in range(N)]
b3 = tracemalloc.get_traced_memory()[0]
tracemalloc.stop(); del bulk_s
print(f"bulk N={N}: dict={(b1-b0)/N:.1f} B/inst  slots={(b3-b2)/N:.1f} B/inst  ratio={(b1-b0)/(b3-b2):.2f}")
PY
```

Two things break the naive "slots always save 6x" story. First, the dict-backed total is flat
at 352 bytes from 1 attribute all the way to 20, because CPython pre-sizes the instance's
key-sharing dictionary to a small quantized capacity and only grows it once enough attributes
force a resize — here, between 20 and 30 attributes, where it jumps to 1,640. So the ratio does
not stay near 6; it *shrinks toward 1 as attribute count grows*, then jumps back up at the next
resize boundary. Second, the bulk cross-check tells a different number from the single-instance
one on purpose: at 3 attributes, `getsizeof` says dict costs 6.29x a slotted instance, but
allocating 50,000 real instances and dividing the traced memory delta by 50,000 gives dict =
104.9 bytes/instance versus slots = 64.9, a ratio of only 1.62. The gap exists because
`sys.getsizeof(obj.__dict__)` counts the full keys table for *every* instance's dict, even
though CPython shares that keys table across instances of the same class (PEP 412) — so summing
per-instance `getsizeof` values double-counts memory that bulk allocation shows is actually
shared. Neither number is wrong; they answer different questions, and only the bulk one answers
"what does adding one more instance cost."

## Practise it

```bash
mlsys grade pyt-getsizeof-one-dict-instance-vs-one-slotted-instance
```

[That task](../tasks/pyt-getsizeof-one-dict-instance-vs-one-slotted-instance/task.md) gates
`size_ratio >= 0.99` against a live CPython oracle that builds the same three-attribute pair
this page measures. The shipped starter is a bare `raise NotImplementedError`, so it fails
immediately; the real trap is returning a plausible-looking hardcoded ratio like `6.0` instead
of measuring `getsizeof` on freshly constructed instances — the gate compares against whatever
your specific interpreter build actually reports, not a textbook number.

In roughly increasing difficulty:
[predict dict and weakref support](../tasks/pyt-predict-dict-and-weakref-support/task.md) (no code),
[the weak-reference slot classifier](../tasks/pyt-weakref-slot-requirement-classifier/task.md),
[the dict-vs-slots footprint ratio](../tasks/pyt-dict-vs-slots-per-instance-footprint-ratio/task.md),
[fix a slots-defeating subclass](../tasks/pyt-fix-a-slots-defeating-subclass/task.md), and
[hand-implement slot descriptors](../tasks/pyt-hand-implement-slot-descriptors/task.md), which
has you write the `__get__`/`__set__` machinery this whole page has been describing from the
outside.

## Common mistakes

- **Hardcoding the ratio instead of measuring it.** The table above already shows the true
  ratio ranges from 1.83 to 8.80 depending only on attribute count, on one interpreter build —
  a constant like `4x` is wrong for most of that range and wrong on any other CPython version.
- **Measuring `getsizeof(obj)` alone for a dict-backed instance.** That reports 56 bytes
  regardless of attribute count, because it excludes the dictionary the object points to —
  missing 296 of the 352 real bytes at 3 attributes.
- **Assuming a subclass inherits the memory saving.** [A subclass that forgets its own
  `__slots__`](../tasks/pyt-fix-a-slots-defeating-subclass/task.md) gets a full `__dict__` back
  silently; nothing warns you, and nothing in the subclass body looks wrong.
- **Forgetting `__weakref__`.** A slotted class cannot be weakly referenced unless
  `"__weakref__"` is explicitly one of the declared slots — costing one more slot to buy back a
  capability dict-backed classes get for free.
- **Reading the single-instance ratio as the real savings at scale.** The bulk cross-check above
  measures 1.62x, not 6.29x, once CPython's shared key-table optimization is allowed to apply
  across many instances of the same class.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[Python Morsels](https://www.pythonmorsels.com/exercises/paths/)** — Trey Hunner's paid
  weekly-exercise subscription has a dedicated descriptors/`__slots__` learning path with an
  explanatory article plus small locally-run test files. The closest topical match found
  anywhere for this exact subject; paid, and it does not measure bytes.
- **[Advanced Python Mastery (David Beazley)](https://github.com/dabeaz-course/python-mastery)**
  — the "Inside Python Objects" section covers slots and descriptors almost one-to-one with this
  page's mechanism, free, with full worked solutions to compare against by hand. No automated
  grading at all.
- **[Fluent Python, 2nd ed.](https://github.com/fluentpython/example-code-2e)** — Luciano
  Ramalho's book is organized explicitly around the data model, including a chapter built
  around `__slots__`, with runnable example code per chapter. Reading and examples only, nothing
  to submit.
- **[Exercism's Python track](https://exercism.org/tracks/python)** — real auto-grading exists,
  but its descriptor-adjacent content is a handful of exercises inside a much larger
  general-Python track, with no dedicated `__slots__` coverage found.
- **[wtfpython](https://github.com/satwikkansal/wtfpython)** — good for the surrounding
  gotchas (identity, interning, mutable defaults) in guess-then-reveal form, but it does not
  touch instance layout or `__slots__` specifically.

## References

1. Python documentation, *Data model* — `object.__slots__`.
   https://docs.python.org/3/reference/datamodel.html#slots
2. Python documentation, `sys.getsizeof`.
   https://docs.python.org/3/library/sys.html#sys.getsizeof
3. Shannon, M. et al., CPython `Objects/dictobject.c` — key-sharing dictionary implementation
   (PEP 412). https://github.com/python/cpython/blob/main/Objects/dictobject.c
