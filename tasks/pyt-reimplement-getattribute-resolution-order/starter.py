from __future__ import annotations


def resolve(obj, name: str):
    """Reimplement `object.__getattribute__`'s default attribute lookup
    order, by hand.

    Parameters
    ----------
    obj : object
        Any instance.
    name : str
        Attribute name to resolve.

    Returns
    -------
    The resolved attribute value, following CPython's real precedence:

      1. A DATA descriptor found by walking `type(obj).__mro__`
         (an object whose type has both `__get__` and `__set__` or
         `__delete__`) -- call its `__get__(obj, type(obj))`.
      2. Otherwise, `obj.__dict__[name]` if present.
      3. Otherwise, a NON-DATA descriptor found in the MRO (`__get__`
         only) -- call its `__get__(obj, type(obj))`.
      4. Otherwise, a plain (non-descriptor) attribute found in the MRO
         -- return it directly.
      5. Otherwise, if some class in the MRO defines `__getattr__`,
         call it as `__getattr__(obj, name)`.
      6. Otherwise, raise `AttributeError(name)`.

    Constraints
    -----------
    Do NOT call `getattr`, `hasattr(obj, name)`, `super()`, or
    `obj.__getattribute__` / `object.__getattribute__` to resolve
    `name` -- that would just delegate the whole problem back to the
    real machinery you're supposed to be reimplementing. Walk
    `type(obj).__mro__` and `klass.__dict__` / `obj.__dict__` directly,
    and check for the descriptor protocol on the ATTRIBUTE'S TYPE
    (e.g. `hasattr(type(value), '__get__')` is fine -- that's checking
    a different name than `name`, and is unavoidable in pure Python).
    """
    raise NotImplementedError('your code here')
