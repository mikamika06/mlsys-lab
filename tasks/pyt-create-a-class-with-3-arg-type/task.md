## Context

The `class` statement is syntactic sugar. Under the hood, defining

```python
class Animal:
    kind = "animal"
```

compiles the class body to a namespace dict and then calls the metaclass —
`type` by default — exactly the way you could call it yourself:

$$
\texttt{Animal} = \texttt{type}(\texttt{"Animal"},\ (\texttt{object},),\ \{\texttt{"kind"}: \texttt{"animal"}, \dots\})
$$

`type(name, bases, namespace)` is the **3-argument form** of `type`: given a
class name, a tuple of base classes, and a namespace `dict` (attributes and
functions, exactly what would have been the class body's local namespace),
it builds and returns a brand-new class object — with a real `__mro__`, real
bound methods, real inheritance. Nothing about the resulting class reveals
whether it came from `class ...:` or from calling `type(...)` directly; they
are the same operation.

## Task

Implement:

```python
def build_animal_hierarchy() -> tuple:
    ...
```

Using **only the 3-argument `type(name, bases, namespace)` call** — no
`class` statement anywhere in your implementation — construct and return a
2-tuple `(Animal, Dog)` of freshly built class objects with:

* `Animal` — base `(object,)`. Namespace has a class attribute
  `kind = "animal"` and a method `speak(self)` returning
  `f"{self.kind} makes a sound"`.
* `Dog` — base `(Animal,)` (the very `Animal` object you just built, so
  `Dog.__mro__` really is `(Dog, Animal, object)`). Namespace overrides
  `kind = "dog"` and overrides `speak(self)` to return `f"{self.kind} barks"`.

## Example

```python
Animal, Dog = build_animal_hierarchy()

a = Animal()
a.speak()          # "animal makes a sound"

d = Dog()
d.speak()           # "dog barks"
isinstance(d, Animal)   # True
Dog.__bases__            # (Animal,)
```

## What the gate checks

The grader defines its own reference `Animal`/`Dog` with an ordinary `class`
statement (the real oracle for "what this hierarchy should look like and
do"), then computes a **structural fingerprint** of both your classes and its
reference classes: the `__mro__` name sequence, the class's own non-dunder
namespace keys, and behavioral output — instantiating each class and reading
`.kind` and calling `.speak()`. It also checks that your `Dog.__bases__[0]`
*is* the exact `Animal` object you returned (not a lookalike). All of this is
compared with `exact_match`, which is `1.0` only if every one of these checks
passes for both classes, and `0.0` otherwise.
