## Context

CPython's compiler runs a constant-folding pass over the AST before
generating bytecode: if every operand of an expression is already a
literal, and evaluating the operation at compile time is both cheap and
safe, the whole expression is replaced by its precomputed value — one
`LOAD_CONST` instead of instructions that rebuild it every call. But the
folder is deliberately conservative. It skips folding when:

* the operation could **raise** at runtime (e.g. a literal division, since
  the compiler won't risk turning a compile-time step into a crash, or
  silently pre-computing something whose failure should surface later),
* the result could be **very large** (e.g. `2 ** 1000` — folding it would
  bloat every compiled copy of the code with a giant embedded integer),
* the container is **mutable** (`list`/`dict`/`set` literals are always
  built fresh at runtime, even with only literal elements — only the
  immutable `tuple` literal gets folded), or
* the expression isn't a constant to begin with (comparisons, function
  calls, or anything touching a name that isn't a literal).

## Task

Implement `predict_folded`:

```python
def predict_folded(exprs):
    """For each source expression string in `exprs`, return True if
    compiling it folds to a single constant, False otherwise."""
```

For each string in `exprs`, determine — by actually compiling it and
inspecting the resulting bytecode, not by static guessing — whether
CPython's constant folder reduced it to a single constant load. A clean way
to do this: compile `"lambda: " + expr` with `compile(..., "eval")` and
`eval` it to get a function, then walk `dis.get_instructions(fn)` (ignoring
the `RESUME` prologue instruction). The expression folded if the remaining
instructions are exactly `LOAD_CONST` followed by `RETURN_VALUE` — or, on
interpreter versions with a dedicated instruction for this, a single
`RETURN_CONST`.

Return a list of `bool`, one per input expression, in the same order.

## Example

```python
predict_folded(["1 + 2", "[1, 2, 3]", "1 / 0"])
# -> [True, False, False]
#    "1 + 2"     -> a single LOAD_CONST 3           (folded)
#    "[1, 2, 3]" -> BUILD_LIST + friends at runtime  (mutable container)
#    "1 / 0"     -> LOAD_CONST 1; LOAD_CONST 0; BINARY_OP  (would raise —
#                   the compiler refuses to fold it away)
```

## What the gate checks

The grader runs your function on a fixed list of 15 expressions — spanning
folded arithmetic, folded string ops, a folded tuple literal, and
deliberately-not-folded cases (division by a literal zero, a huge power, a
list/dict/set literal, a comparison, and a name reference) — and compares
your 15 booleans element-by-element against the same live-compile-and-`dis`
procedure applied independently inside the grader. **exact_match** is `1.0`
only if every single prediction matches; any mismatch (or an exception)
makes it `0.0`.
