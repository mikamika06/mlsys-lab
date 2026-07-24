## Context

Python closures store information about lexical scope in code objects. A variable
defined in an outer function and referenced by an inner function becomes a
cell variable of the outer code object. The corresponding name appears as a
free variable in the inner code object.

For a nested function relationship, the compiler records two sets:

$$
\mathrm{co\_cellvars} = \{\text{names captured by child scopes}\}
$$

and

$$
\mathrm{co\_freevars} = \{\text{names required from parent scopes}\}.
$$

These attributes are created by CPython during compilation and are available on
code objects.

## Task

Implement `analyze_closure(source)`:

```python
def analyze_closure(source: str) -> dict:
    ...
```

The input is a Python source string. Compile it and inspect every code object
inside the compiled module. Return a dictionary mapping each code object's
qualified name to its closure classification:

```python
{
    "outer": {
        "co_cellvars": ["x"],
        "co_freevars": []
    },
    "outer.inner": {
        "co_cellvars": [],
        "co_freevars": ["x"]
    }
}
```

The function must include the module code object under the key `"module"`.
Nested functions should use dot-separated names based on their lexical nesting.
The lists must be sorted alphabetically.

Use Python code object attributes such as `co_cellvars` and `co_freevars`.
Do not execute the input source.

## Example

```python
source = """
def make_adder():
    x = 10
    def add(y):
        return x + y
    return add
"""

result = analyze_closure(source)

# result["make_adder"]["co_cellvars"] == ["x"]
# result["make_adder.add"]["co_freevars"] == ["x"]
```

## What the gate checks

The gate builds several source programs containing nested functions and computes
the expected answer from real CPython code objects produced by `compile()`.
Your implementation must exactly match the compiler-reported
`co_cellvars` and `co_freevars` values for every discovered code object.

The output is checked with exact equality.
