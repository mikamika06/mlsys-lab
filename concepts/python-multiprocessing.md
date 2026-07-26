---
title: "What is python multiprocessing?"
description: "Python multiprocessing explained, with a measured bytes-pickled-per-argument-shape table you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is python multiprocessing?

Python multiprocessing is the standard-library module that runs Python code across separate
OS processes instead of threads, giving each process its own interpreter and memory so
CPU-bound work escapes the GIL entirely. The cost of that isolation is whatever has to cross
the process boundary: pickling a plain 1,000,000-element list of ints costs 4,871,352 bytes,
while the same numbers as a shared-memory descriptor cost 67. Below is exactly what gets
pickled, and what refuses to pickle at all, for the shapes an argument commonly takes.

## How it works

A `Process` — and everything built on it, `Pool`, `Queue`, `Manager` — is not a lighter
thread with its own lock. It is a second, independent interpreter with its own heap, its own
GIL, and its own copy of whatever the parent had at the moment the child was created. How the
child gets "whatever the parent had" is controlled by the *start method*, and the three
methods CPython ships disagree about it completely. `fork`, Linux's default, clones the
parent's entire address space with copy-on-write — the child already has every object,
function, and closure the parent had, byte-for-byte, without one call to `pickle`. `spawn`,
the default on macOS since Python 3.8 and the only method Windows has ever had (Windows has
no `fork()` syscall), starts a brand-new interpreter from scratch and re-imports only
`__main__`; nothing carries over automatically, so the target callable and every argument
must travel through `pickle` to reach it. `forkserver` is a middle path, forking from a
clean, pre-warmed helper.

macOS's move to `spawn` was not a style choice: forking a process that already has Objective-C
frameworks loaded is documented as unsafe, since those frameworks can start their own threads
that `fork()` does not carry into the child, leaving locks those threads held permanently
stuck. That default is why this page's subject is a correctness question, not a speed one.
Under `fork`, a closure passed as a `Pool` target works, since the child already holds it in
memory and only the call arguments need pickling. Run the identical script under `spawn` —
the default most contributors are actually on — and the same closure fails immediately,
because `pickle`'s function reducer stores nothing but a dotted `module.qualname` for the
child to re-import, and a function defined inside another has no such name.

The same boundary applies to every argument and return value, not only the callable. An
instance layout that is cheap to build, like the fixed slots in
[python slots](python-slots.md), still has to be flattened into bytes to cross a pipe, and a
NumPy array pays that cost as one memcpy-shaped blob; [memory coalescing](memory-coalescing.md)
and [cache blocking](cache-blocking.md) ask the same "what has to move, in what shape" question
one level down, on a bus instead of a pipe. The payoff of measuring this is workload
selection: many random restarts of [k-means](kmeans.md) or independent [PCA](pca.md) fits per
shard are exactly the embarrassingly-parallel shape `Pool.map` was built for, provided what
each worker gets is cheap to serialize.

## Bytes pickled measured against argument shape

The script below builds the same logical payload — 1,000,000 integers — in six shapes and
pickles each with protocol 5, the protocol PEP 574 added out-of-band buffer support to. It
also pickles a closure and a top-level function, since callables cross the same boundary as
data and obey the same rules.

| argument shape (1,000,000 elements) | bytes pickled (protocol 5) | spawn-safe |
|---|---|---|
| Python list of ints | 4,871,352 | yes |
| NumPy int64 array, in-band | **8,000,139** | yes |
| NumPy int64 array, out-of-band buffer | 121 (+ an 8,000,000-byte buffer kept outside the pickle stream) | yes, but `Pool` won't wire this up for you |
| `shared_memory` descriptor (name + shape + dtype) | 67 | yes |
| generator yielding the same 1,000,000 ints | `TypeError: cannot pickle 'generator' object` | no |
| closure over a local variable | `AttributeError: Can't pickle local object 'make_closure.<locals>.inc'` | no |
| top-level function | 37 | yes |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import pickle, multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

N = 1_000_000

lst = list(range(N))
arr = np.arange(N, dtype=np.int64)
buffers = []
oob = pickle.dumps(arr, protocol=5, buffer_callback=buffers.append)

shm = shared_memory.SharedMemory(create=True, size=N * 8)
descriptor = {"name": shm.name, "shape": arr.shape, "dtype": arr.dtype.str}

def make_closure():
    total = 0
    def inc(x):
        return x + total
    return inc

def top_level(x):
    return x + 1

def pickled_or_error(obj):
    try:
        return str(len(pickle.dumps(obj, protocol=5)))
    except (TypeError, AttributeError) as e:
        return f"{type(e).__name__}: {e}"

print("start_method:", mp.get_start_method())
print("list_bytes:", len(pickle.dumps(lst, protocol=5)))
print("numpy_inband_bytes:", len(pickle.dumps(arr, protocol=5)))
print("numpy_oob_bytes:", len(oob), "oob_buffers:", len(buffers), "oob_buffer_nbytes:", buffers[0].raw().nbytes)
print("shm_descriptor_bytes:", len(pickle.dumps(descriptor, protocol=5)))
print("generator:", pickled_or_error(x for x in range(N)))
print("closure:", pickled_or_error(make_closure()))
print("top_level_fn_bytes:", len(pickle.dumps(top_level, protocol=5)))

shm.close()
shm.unlink()
PY
```

The ordering matters more than the spread: the plain Python list (4,871,352 bytes) is
smaller than the in-band NumPy array (8,000,139 bytes), because a small Python int pickles at
roughly 4.9 bytes while a fixed int64 element always costs 8 — "just use NumPy" only shrinks
the pickle once values stop being small integers, or once out-of-band buffers replace the
in-band path. Out-of-band gets the array down to 121 bytes on the wire, but
`multiprocessing`'s own pickler, `ForkingPickler.dumps`, calls `dump()` with no
`buffer_callback`, so a `Pool.map` argument that is a NumPy array is copied in-band on every
call unless you build the channel yourself. `shared_memory` is the version of that saving
`multiprocessing` gives you for free: put the array in a named shared block once, and every
worker after that receives a 67-byte descriptor instead of a fresh 8-megabyte copy. A
generator or a closure do not shrink under any trick — they refuse to serialize at all,
`TypeError` and `AttributeError` respectively, a louder failure than a silent size difference
would be.

## Practise it

```bash
mlsys grade pyt-reconstruct-closure-cells
```

No task in this bank grades bytes pickled across a real process boundary — the module is
cross-platform-flaky to grade deterministically, since which start methods exist differs by
OS. [That task](../tasks/pyt-reconstruct-closure-cells/task.md) is the closest relative
instead: gated on `exact_match == 1.0`, it makes you read `fn.__code__.co_freevars` and
`fn.__closure__` by hand and return each `(name, value)` pair — the machinery that decides why
the closure row above fails to pickle while the top-level-function row does not. The shipped
starter is a bare `raise NotImplementedError('your code here')`; the real trap once you
replace that is pairing the two sequences positionally instead of checking they are the same
length, since neither carries the other's names.

More on the same boundary, in roughly increasing difficulty:
[classify closure cellvars and freevars](../tasks/pyt-cellvars-vs-freevars/task.md),
[generator vs list comprehension footprint](../tasks/pyt-generator-vs-list-comp-footprint/task.md)
(the `sys.getsizeof` half of the story above, without the pickle half),
[classify workloads that scale with threads under the GIL](../tasks/pyt-scales-with-threads-classifier/task.md)
(why you would reach for processes over threads at all), and
[lazy pipeline processes only K of N](../tasks/pyt-lazy-pipeline-processes-only-k-of-n/task.md)
(the generator side of the row that refuses to pickle above).

## Common mistakes

- **Passing a NumPy array through `Pool.map` and expecting `shared_memory`-level cost.**
  Nothing in stock `multiprocessing` routes an array through PEP 574's out-of-band path;
  every call pays the in-band cost above — 8,000,139 bytes copied into the pickle stream, and
  as much again on the way back if the worker returns it.
- **Using a closure or a `lambda` as a `Process`/`Pool` target.** It works under `fork`
  (Linux's default) since the child already has it in memory, and fails immediately with
  `AttributeError: Can't pickle local object 'make_closure.<locals>.inc'` under `spawn` — the
  default on macOS since Python 3.8 and the only option on Windows — so code that passes on a
  Linux CI box breaks on a contributor's Mac.
- **Passing an unconsumed generator as a worker argument.** It is not silently converted to a
  list; `pickle` has no reducer for generator objects, so it raises
  `TypeError: cannot pickle 'generator' object` before any work starts.
- **Assuming `shared_memory` removes the need for synchronization, not just the copy.** It
  cuts the transfer from 8,000,000 bytes to a 67-byte descriptor, but workers writing the same
  block concurrently still need a `Lock` — the descriptor says nothing about who may write
  where.
- **Materializing a lazy pipeline before handing it to a worker "to be safe."**
  `list(generator)` turns something that could stream through a pipe into a payload the size
  of the list row above, for no reason the target process needed.

## Where else to practise this

None of the resources surveyed for this track single out multiprocessing or pickling-as-IPC
— the [full survey of what exists](../LANDSCAPE.md) already concludes this area's "C-runtime
half" (GIL, refcounting, bytecode) has almost no gradable practice anywhere, and that holds
here too.

- **[Exercism — Python track](https://exercism.org/tracks/python)** — real auto-grading
  across 146 exercises, but no multiprocessing or IPC content; its coverage of this area is a
  handful of OOP/data-model concepts.
- **[Python Morsels](https://www.pythonmorsels.com/exercises/paths/)** — paid, with dedicated
  paths for descriptors, metaclasses, and generators-and-iterators, but no concurrency or
  multiprocessing path.
- **[wtfpython](https://github.com/satwikkansal/wtfpython)** — free, and its gotcha catalogue
  covers GIL/threading surprises, but guess-then-reveal reading, not graded, and about threads
  rather than processes.
- **[Python behind the scenes, post #13](https://tenthousandmeters.com/blog/python-behind-the-scenes-13-the-gil-and-its-effects-on-python-multithreading/)**
  — the closest piece of writing found to this page's motivation (why the GIL pushes CPU-bound
  work toward processes), but read-only, about threads not the module itself, and the site did
  not respond when last checked.
- **[Advanced Python Mastery (David Beazley)](https://github.com/dabeaz-course/python-mastery)**
  — free, full worked solutions, closest topical match here for object/iterator internals
  generally, but no multiprocessing-specific material found in it.

## References

1. Python documentation, *multiprocessing* — "Contexts and start methods".
   https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
2. Python documentation, *pickle* — "Out-of-band buffers" (PEP 574).
   https://docs.python.org/3/library/pickle.html#out-of-band-buffers
3. Python documentation, *multiprocessing.shared_memory*.
   https://docs.python.org/3/library/multiprocessing.shared_memory.html
