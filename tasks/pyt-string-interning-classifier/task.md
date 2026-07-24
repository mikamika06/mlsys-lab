## Context

Python strings are immutable objects. CPython may reuse a single object for multiple equal strings through string interning. When two references point to the same object, the identity operator `is` is true.

String equality checks values:

$$a == b$$

while identity checks whether two references have the same object:

$$a \ \mathrm{is}\ b$$

Interning is an implementation detail used by CPython for some strings, especially identifier-like literals. A runtime operation that creates a new string object may or may not share the same object as an existing literal.

## Task

Implement `classify_interning(pairs)`:

```python
def classify_interning(pairs):
    ...
```

The input is a list of pairs. Each pair contains two Python strings. Return a list of booleans where element $i$ is `True` exactly when the two strings in input pair $i$ are the same object according to Python identity (`is`), and `False` otherwise.

Do not compare only string contents. The task is about CPython object identity and interning behavior.

## Example

```python
pairs = [
    ("identifier_name", "identifier_name"),
    ("identifier_name", "".join(["identifier", "_name"]))
]

result = classify_interning(pairs)
# [True or False, True or False depending on CPython identity]
```

The exact values above are not fixed by the example. The implementation must inspect object identity.

## What the gate checks

The gate creates string pairs using real CPython string construction behavior and computes the expected result with the identity operation itself.

The returned list must exactly match the oracle result:

$$\mathrm{expected}_i = (a_i \ \mathrm{is}\ b_i)$$

The gate rejects solutions that use value equality (`==`) because equal strings can be different objects.
