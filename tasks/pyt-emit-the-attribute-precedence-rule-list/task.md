## Context

Python attribute lookup combines several mechanisms: instance dictionaries, descriptors, class attributes, and fallback hooks.

For an object $o$ and attribute name $n$, lookup chooses between possible providers according to a precedence relation:

$$
\text{data descriptor} \succ \text{instance dictionary} \succ \text{non-data descriptor} \succ \text{class attribute} \succ \text{__getattr__ fallback}.
$$

The method `__getattribute__` is called before normal attribute lookup begins. If the normal lookup path cannot find an attribute, `__getattr__` may provide a fallback value.

A descriptor defining `__get__` participates in attribute resolution. A data descriptor additionally defines `__set__` or `__delete__`, which gives it priority over entries stored in the instance dictionary.

This task encodes the observed precedence chain as integers:

- `1`: data descriptor
- `2`: instance dictionary
- `3`: non-data descriptor
- `4`: class attribute
- `5`: `__getattribute__`
- `6`: `__getattr__`

## Task

Implement `emit_attribute_precedence_rule_list()`:

```python
def emit_attribute_precedence_rule_list() -> list[int]:
    ...
```

Return the integer-encoded ordered sequence for Python attribute lookup precedence. The sequence must include both lookup hooks and the normal providers.

The return value must be a list of integers. Do not return strings, sets, or explanatory text.

## Example

```python
chain = emit_attribute_precedence_rule_list()

# Expected shape:
# [5, 1, 2, 3, 4, 6]
```

The exact sequence is checked against a reference produced from real CPython attribute behavior.

## What the gate checks

The gate builds instrumented classes with real descriptors, real instance dictionaries, and real fallback hooks. It derives the precedence order by observing which provider wins in actual attribute accesses.

The `exact_match` metric must equal $1.0$.
