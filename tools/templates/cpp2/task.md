## Context

A C compiler lays out a struct's fields in order, inserting **padding** so each
field starts at a multiple of its alignment, and pads the whole struct to its
largest field's alignment. Under the pinned LP64 ABI: `char`=1, `short`=2,
`int`=4, `long`/`double`/pointer=8; alignment equals size.

## Task

Implement `struct_size(fields)` returning `sizeof` of a struct with those fields,
computing the padding yourself.

## Example

```python
struct_size(["char", "int", "double"])  # -> 16  (offsets 0, 4, 8)
```

## What the gate checks

Your size must equal the pinned-ABI reference (`arena.cppabi`) for several field
lists ($\mathrm{exact\_match}=1$).
