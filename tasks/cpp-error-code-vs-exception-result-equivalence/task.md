## Context

C++ offers multiple models for error handling. Two prominent ones:

1. **Exceptions**: when an error occurs, an exception is `throw`n and the
   stack unwinds until a matching `catch` is found. The "happy path"
   carries no error-checking branches.
2. **Error codes / `std::expected`**: errors come back as ordinary
   values (e.g. a discriminated union like `std::expected<T, E>`). The
   caller must explicitly check the return value.

To model `std::expected<double, int>` without a real union, consider this
naive tagged struct:

```cpp
struct NaiveExpected {
    bool   has_value;
    double val;
    int    err;
};
```

## Task

Implement, in `solve.cpp`, all three functions declared in `sol.hpp`:

```cpp
NaiveExpected compute_expected(const std::vector<std::string>& ops);
double        compute_exceptions(const std::vector<std::string>& ops);
long          naive_expected_size();
```

Both `compute_expected` and `compute_exceptions` run the same op sequence
against a `double state` starting at `0.0`:

- `"add X"` → `state += X`
- `"sub X"` → `state -= X`
- `"div0"` → fails with error code `1`
- `"overflow"` → fails with error code `2`

`compute_expected` (error-code model):

- On success (every op consumed without failing): return
  `{has_value: true, val: final state}`.
- On failure: return `{has_value: false, err: code}` immediately — ops
  after the failing one are never applied.

`compute_exceptions` (exception model), using **real** C++
`throw`/`catch`:

- On success: return the final `state`.
- On failure: `throw OpFailure(code)` immediately (`OpFailure` is
  declared in `sol.hpp`) — ops after the failing one are never applied.

`naive_expected_size`: return `sizeof(NaiveExpected)` as the real compiler
lays it out.

## Example

For `ops = {"add 2.0", "div0"}`:

- `compute_expected(ops)` returns `{has_value: false, err: 1}`.
- `compute_exceptions(ops)` throws `OpFailure(1)`.

For `ops = {"add 1.5", "add 2.5"}` (no failure):

- `compute_expected(ops)` returns `{has_value: true, val: 4.0}`.
- `compute_exceptions(ops)` returns `4.0`.

## What the gate checks

The fixed driver (`main.cpp`) runs five fixed op sequences through both
models, prints `naive_expected_size()` once, then for each sequence
prints `compute_expected`'s result and either `compute_exceptions`'s
returned value or the `OpFailure` it actually caught via a real
`try`/`catch`. The gate is an exact string match (`exact_match == 1.0`)
against the reference's printed output: the two models must agree with
each other and with the reference on every case, including which
operations get applied before a failure stops the sequence.
