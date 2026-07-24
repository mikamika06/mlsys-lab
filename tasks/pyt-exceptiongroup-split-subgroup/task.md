## Context

Python 3.11 introduced exception groups, which allow multiple exceptions to be
raised and handled together while preserving their nested structure. An
`ExceptionGroup` can contain ordinary exceptions and nested exception groups.

The `split` method partitions an exception group into two related groups:

$$
\mathrm{split}(E, P) = (E_{\mathrm{match}}, E_{\mathrm{rest}})
$$

where leaves satisfying predicate $P$ appear in the matched subgroup and all
other leaves appear in the remaining subgroup. The resulting groups preserve
the original nesting shape and metadata.

For example, splitting by exception type should not flatten a tree. A nested
group containing a nested subgroup should produce nested subgroups in the
outputs.

## Task

Implement `split_group(eg, names)`.

The function receives:

- `eg`: an `ExceptionGroup` instance.
- `names`: an iterable of exception class names such as `("ValueError", "KeyError")`.

Return a pair `(matched, rest)` by using `ExceptionGroup.split(predicate)`.
The predicate should select leaf exceptions whose class name is present in
`names`.

The returned values must preserve the nested exception group structure created
by Python. Do not flatten the group or rebuild it manually from leaf exceptions.

The function must work when either side of the split is `None`.

## Example

```python
eg = ExceptionGroup(
    "root",
    [
        ValueError("bad"),
        ExceptionGroup("nested", [TypeError("wrong"), KeyError("missing")]),
    ],
)

matched, rest = split_group(eg, ("ValueError", "KeyError"))
```

The matched result contains the `ValueError` and the nested `KeyError` while the
rest result contains the nested `TypeError`. The inner `nested` group remains a
nested group in both outputs where applicable.

## What the gate checks

The gate builds several real `ExceptionGroup` instances and computes the
expected result using Python's own `ExceptionGroup.split` implementation. The
candidate output is converted to a structural representation containing group
messages and exception class names.

The `exact_match` score is $1.0$ only when the candidate has exactly the same
nested structure as the CPython result.
