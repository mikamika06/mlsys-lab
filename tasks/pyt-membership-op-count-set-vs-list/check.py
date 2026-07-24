class _Probe:
    calls = 0

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        type(self).calls += 1
        return isinstance(other, _Probe) and self.value == other.value


def _oracle(keys, queries):
    values = [_Probe(x) for x in keys]
    list_values = list(values)
    set_values = set(values)

    _Probe.calls = 0
    for q in queries:
        _Probe(q) in list_values
    list_count = _Probe.calls

    _Probe.calls = 0
    for q in queries:
        _Probe(q) in set_values
    set_count = _Probe.calls

    return list_count, set_count


def grade(sol, fx) -> dict:
    cases = [
        (list(range(8)), [7, 8, 3, 20]),
        (list(range(100)), [99, 100, -1, 50]),
        ([1, 5, 9, 13, 17], [17, 2, 13, 17]),
    ]

    list_ok = 1.0
    set_ok = 1.0

    for keys, queries in cases:
        ref_list, ref_set = _oracle(keys, queries)
        try:
            got_list, got_set = sol.membership_op_counts(
                list(keys), list(queries)
            )
        except Exception:
            return {
                "list_eq_match": 0.0,
                "set_eq_match": 0.0,
            }

        if got_list != ref_list:
            list_ok = 0.0
        if got_set != ref_set:
            set_ok = 0.0

    return {
        "list_eq_match": list_ok,
        "set_eq_match": set_ok,
    }
