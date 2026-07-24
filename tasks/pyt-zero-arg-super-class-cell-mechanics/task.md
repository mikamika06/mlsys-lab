## Context

Python 3's zero-argument `super()` looks like magic — how does it know
which class and instance it's operating on, with no arguments at all? The
trick happens at **compile time**, not call time: when the compiler sees a
bare reference to the name `super` (or `__class__`) anywhere in a method
body written inside a `class` statement, it treats `__class__` as an
implicit free variable of that method, exactly as if the method had closed
over a variable named `__class__` from the enclosing class body.

Concretely, after such a method is compiled:

$$
\texttt{"\_\_class\_\_"} \in \texttt{method.\_\_code\_\_.co\_freevars}
$$

and `method.__closure__` contains a real cell at the matching index, whose
`cell_contents` is the class the method was lexically defined in. Zero-arg
`super()` then reads this cell at call time instead of taking `__class__`
as an argument. A method that never mentions `super` or `__class__` gets no
such freevar and no such cell — there is nothing implicit to carry.

## Task

Implement `class_cell_info`:

```python
def class_cell_info(method):
    """Inspect a method's code object for the implicit __class__ closure
    cell that zero-arg super() (and bare __class__) rely on. Returns the
    referenced class's __name__, or None if the method carries no such
    cell."""
```

* `method` — an unbound method / plain function object (e.g. `SomeClass.some_method`).
* If `"__class__"` is present in `method.__code__.co_freevars`, find its
  index and read the corresponding cell out of `method.__closure__`;
  return `cell.cell_contents.__name__`.
* Otherwise, return `None`.

## Example

```python
class Base:
    def greet(self):
        return "base"

class Child(Base):
    def greet(self):
        return super().greet() + "-child"

class_cell_info(Child.greet)   # -> "Child"

class Plain:
    def greet(self):
        return "plain"

class_cell_info(Plain.greet)   # -> None
```

## What the gate checks

The grader builds four methods and compares your function's output against
the same direct introspection, applied independently:

* a subclass method that calls zero-arg `super()` — expects the
  **subclass**'s name (the cell always refers to the class the `def`
  lexically lives in, not any ancestor);
* a plain method that never mentions `super` or `__class__` — expects
  `None`;
* a method that references bare `__class__` directly (no `super()` call at
  all) — the same mechanism still fires, expects that class's name;
* a method using *explicit*-argument `super(Cls, self)` — still expects
  that class's name, since the compiler's detection is based on the name
  `super` appearing in the method body at all, not on which call form is
  used.

**exact_match** is `1.0` only if all four cases match exactly; any mismatch
or exception makes it `0.0`.
