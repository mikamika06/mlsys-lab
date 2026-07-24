def _oracle():
    events = []

    class Descriptor:
        def __init__(self, name):
            self.name = name

        def __set_name__(self, owner, name):
            events.append("set_name:" + self.name)

    class Meta(type):
        @classmethod
        def __prepare__(mcls, name, bases):
            events.append("prepare")
            return {}

        def __new__(mcls, name, bases, namespace):
            events.append("new")
            return super().__new__(mcls, name, bases, namespace)

        def __init__(cls, name, bases, namespace):
            events.append("init")
            super().__init__(name, bases, namespace)

    class Base:
        def __init_subclass__(cls):
            events.append("init_subclass")
            super().__init_subclass__()

    class Fixture(Base, metaclass=Meta):
        first = Descriptor("first")
        second = Descriptor("second")

    return events


def grade(sol, fx) -> dict:
    try:
        got = sol.class_creation_events()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == _oracle() else 0.0}
