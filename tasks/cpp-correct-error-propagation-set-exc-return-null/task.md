## Context

When writing CPython C extensions, every function exposed to Python must
follow the **error protocol**: if an error occurs, set a Python exception
via `PyErr_SetString` and return `NULL`; if there is no error, return a
valid non-`NULL` `PyObject*` and leave the error indicator unset. This
task models that exact protocol in real, self-contained C++ (declared in
`sol.hpp`, defined in `main.cpp` — you don't touch it), so you can practice
the discipline without needing a CPython build toolchain:

- `ExcState g_exc` plays the role of CPython's per-thread error indicator.
- `set_error(type, msg)` plays the role of `PyErr_SetString`.
- `PyFloatObj*` plays the role of a `PyObject*` returned from
  `PyFloat_FromDouble`; `nullptr` means "an error is set, check `g_exc`".

Two classic bugs break this protocol: returning `NULL` *without* setting an
error (the caller sees "no error, no result" and crashes on the null
dereference), and returning a non-`NULL` value *after* setting an error (the
caller ignores the exception and keeps going with garbage).

## Task

Implement `safe_divide` in `solve.cpp`:

```cpp
PyFloatObj* safe_divide(double a, double b);
```

- If `b == 0.0`: call `set_error(ExcType::ZeroDivisionError, "division by
  zero")` and return `nullptr`. Do not also return an object.
- Otherwise: return `new PyFloatObj{a / b}`, and do not call `set_error` —
  `g_exc` must stay `ExcType::None`.

The fixed driver in `main.cpp` resets `g_exc` before each call, calls
`safe_divide(a, b)` over several `(a, b)` fixtures (including some with
`b == 0`), and prints either the resulting value or the captured exception
name and message.

## Example

```
safe_divide(10.0, 2.0)  -> value=5.000000 exc=None
safe_divide(3.0, 0.0)   -> NULL exc=ZeroDivisionError msg=division by zero
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across seven `(a, b)` fixtures,
several with `b == 0`. The starter always returns `nullptr` without ever
calling `set_error`, so every fixture — valid and invalid alike — prints the
wrong line.
