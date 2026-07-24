def _oracle(method):
    """Real CPython introspection — no separate algorithm, this IS the
    mechanism being tested."""
    freevars = method.__code__.co_freevars
    if "__class__" not in freevars:
        return None
    idx = freevars.index("__class__")
    cell = method.__closure__[idx]
    return cell.cell_contents.__name__


def _make_fixtures():
    class Base:
        def greet(self):
            return "base"

    class Child(Base):
        def greet(self):
            # zero-arg super() -> should carry a __class__ cell for Child
            return super().greet() + "-child"

    class Plain:
        def greet(self):
            # no super, no __class__ -> no cell at all
            return "plain"

    class Direct:
        def whoami(self):
            # bare __class__ reference (no super() call) also triggers it
            return __class__.__name__

    class Grandparent:
        def greet(self):
            return "grandparent"

    class ExplicitSuper(Grandparent):
        def greet(self):
            # even explicit-argument super() references the name "super",
            # so the compiler still adds the __class__ freevar
            return super(ExplicitSuper, self).greet() + "-explicit"

    return {
        "Child.greet": (Child.greet, "Child"),
        "Plain.greet": (Plain.greet, None),
        "Direct.whoami": (Direct.whoami, "Direct"),
        "ExplicitSuper.greet": (ExplicitSuper.greet, "ExplicitSuper"),
    }


def grade(sol, fx) -> dict:
    fixtures = _make_fixtures()
    ok = 1.0
    for _, (method, expected) in fixtures.items():
        try:
            got = sol.class_cell_info(method)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
