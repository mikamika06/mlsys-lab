## Context

`import`ing a module in CPython does two things: (1) if the module isn't
already in `sys.modules`, run its body top-to-bottom, populating its
namespace as it goes, and register it in `sys.modules` **before**
running the body (not after); (2) bind a name in the *importing*
module's namespace.

That "register before running" step is exactly what makes circular
imports sometimes work and sometimes explode. If module `p` is mid-way
through its body and hits `import q`, and `q`'s body in turn does
`import p`, Python sees `p` already sitting in `sys.modules` — so it
does **not** re-run `p`'s body (that would infinite-loop). It just hands
back `p`'s namespace **as far as it has gotten so far**. Whether that's
a problem depends entirely on *what kind* of import statement is used
and *how much* of `p` had executed before the cycle was triggered:

* `import q` never inspects `q`'s namespace — it just binds a reference
  to whatever module object `q` currently is, complete or not. This
  statement, by itself, can **never** raise from circularity.
* `from q import name` **does** look up `name` in `q`'s namespace right
  then. If `q` is mid-execution and hasn't reached the statement that
  defines `name` yet, this raises
  `ImportError: cannot import name 'name' from partially initialized module 'q' (most likely due to a circular import)`.

## Task

You're given a tiny, restricted import graph — no packages, no relative
imports, just top-level modules whose bodies are a short ordered list of
one of three statement kinds:

```python
("bind", name)                 # name = 1               (defines `name`)
("import_module", other)       # import other
("from_import", other, name)   # from other import name
```

Implement `predict_import_result`:

```python
def predict_import_result(modules: dict, entry: str) -> bool:
    ...
```

* `modules` — `{module_name: [ops...]}` as above.
* `entry` — the module name that gets imported first.
* Returns `True` if the whole import completes without error, `False`
  if it raises `ImportError`.

You must reason about this **statically** (simulate the import
machinery yourself) — do not literally write files and exec them.

## Example

```python
modules = {
    "p": [("from_import", "q", "y")],
    "q": [("import_module", "p"), ("bind", "y")],
}
predict_import_result(modules, "p")   # -> True
predict_import_result(modules, "q")   # -> False
```

The same graph, two different outcomes, depending only on which module
starts the chain:

* **`entry="p"`**: importing `p` runs `from q import y`, so Python
  imports `q` from scratch. `q`'s body runs `import p` first — but `p`
  is already in `sys.modules` (registered the moment we started
  importing it, before running its body), so this just binds a
  reference to `p`'s still-empty namespace; no error. `q` then runs
  `bind y` and finishes completely. Control returns to `p`'s
  `from q import y`: `q` is now fully done and `y` **is** defined there
  — succeeds.
* **`entry="q"`**: importing `q` runs `import p` first, so Python
  imports `p` from scratch. `p`'s body immediately runs
  `from q import y` — but `q` is already in `sys.modules` (registered,
  still stuck on its very first statement), and `y` has **not** been
  bound yet (that's `q`'s *second* statement, not reached). This raises
  `ImportError`.

## What the gate checks

**exact_match** — 24 fixed import-graph fixtures (both `bool` labels
represented). For every fixture, the grader compiles the graph to real
`.py` files and imports them in an actual, fresh CPython subprocess,
recording whether `ImportError` was raised — this is the real, live
ground truth, recomputed from scratch on every grading run, never a
hardcoded table. Your `predict_import_result` output must exactly match
that real outcome on all 24 fixtures. Gate `== 1.0`.
