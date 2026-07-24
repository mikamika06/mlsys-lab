## Context

Python's exception hierarchy has a deliberate split at `BaseException`:
`Exception` (and everything a normal program is expected to catch and
handle -- `ValueError`, `KeyError`, a custom validation error, ...) is a
*subclass* of `BaseException`, but a handful of control-flow signals --
`SystemExit`, `KeyboardInterrupt`, `GeneratorExit` -- deliberately live
**outside** `Exception`, precisely so that generic error-handling code does
not accidentally swallow "please stop the program now."

$$
\texttt{BaseException} \supset \texttt{Exception} \supset \{\texttt{ValueError}, \texttt{KeyError}, \dots\}, \qquad
\texttt{SystemExit} \in \texttt{BaseException} \setminus \texttt{Exception}.
$$

A bare `except:` clause catches `BaseException` -- *everything*, including
the signals that were never meant to be caught by ordinary error-handling
code. Combined with a handler that discards the caught object (`except exc:`
never bound, or bound but never inspected), this is one of the most common
real-world bugs: every failure, no matter its real cause, gets reported as
the same generic, wrong sentinel -- and a `SystemExit` meant to actually
stop the program gets silently absorbed instead.

## Task

Implement `classify_failure`:

```python
def classify_failure(fn: Callable[[], Any]) -> str:
    ...
```

* `fn` -- a zero-argument callable.

Call `fn()` and classify what happens:

* If it returns normally, return the string `"OK"`.
* If it raises something that **is an instance of `Exception`**, catch it
  and return `type(exc).__name__` -- the *exact* class name of whatever was
  actually raised (not a generic placeholder).
* If it raises something that is a `BaseException` but **not** an
  `Exception` (a control-flow signal), do **not** catch it -- let it
  propagate out of `classify_failure` unchanged, so the caller still sees
  it.

## Example

```python
def boom():
    raise KeyError("missing")

print(classify_failure(boom))       # "KeyError"
print(classify_failure(lambda: 1))  # "OK"

def stop():
    raise SystemExit(1)

classify_failure(stop)              # raises SystemExit -- NOT caught,
                                     # NOT reported as a string
```

## What the gate checks

A single gate named **exact_match** runs `classify_failure` on a set of
probe functions -- a clean success, several built-in exception types, a
custom `Exception` subclass, an exception raised several call-frames deep,
and a custom `BaseException` subclass that is not an `Exception`. For every
`Exception`-raising or successful case, the real outcome is obtained by
directly executing the same probe function in the grader's own
`try/except` (a real, observed oracle, not a hardcoded string) and compared
against your return value. For the `BaseException`-but-not-`Exception`
case, the gate requires that calling `classify_failure` raises that exact
object back out, rather than swallowing it or reporting it as a string.
Any single mismatch fails the gate.
