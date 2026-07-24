def membership_op_counts(keys, queries):
    class Probe:
        calls = 0

        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return hash(self.value)

        def __eq__(self, other):
            type(self).calls += 1
            return isinstance(other, Probe) and self.value == other.value

    values = [Probe(x) for x in keys]

    Probe.calls = 0
    as_list = list(values)
    for q in queries:
        Probe(q) in as_list
    list_count = Probe.calls

    Probe.calls = 0
    as_set = set(values)
    for q in queries:
        Probe(q) in as_set
    set_count = Probe.calls

    return list_count, set_count
