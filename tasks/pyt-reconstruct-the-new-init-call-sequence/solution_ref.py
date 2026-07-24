def construction_sequence():
    events = []

    class Root:
        def __new__(cls, *args, **kwargs):
            events.append((cls.__name__, "__new__"))
            return super().__new__(cls)

        def __init__(self):
            events.append((self.__class__.__name__, "__init__"))

    class Branch(Root):
        def __new__(cls, *args, **kwargs):
            events.append((cls.__name__, "__new__"))
            return super().__new__(cls)

    class Leaf(Branch):
        def __init__(self):
            events.append((self.__class__.__name__, "__init__"))
            super().__init__()

    Leaf()
    return events
