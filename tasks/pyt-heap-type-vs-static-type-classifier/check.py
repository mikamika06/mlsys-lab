def _oracle(types):
    heap_mask = 1 << 9
    return [bool(t.__flags__ & heap_mask) for t in types]


def grade(sol, fx) -> dict:
    class LocalA:
        pass

    class LocalB:
        pass

    BuiltinNamedHeap = type(
        "BuiltinNamedHeap",
        (),
        {"__module__": "builtins"},
    )

    cases = [
        [
            int,
            str,
            dict,
            list,
            tuple,
            object,
        ],
        [
            LocalA,
            LocalB,
            type,
            Exception,
            ValueError,
            BuiltinNamedHeap,
        ],
        [
            bytes,
            float,
            complex,
            set,
            frozenset,
            property,
            staticmethod,
            classmethod,
        ],
    ]

    ok = 1.0
    for types in cases:
        try:
            got = sol.classify_heap_types(types)
        except Exception:
            ok = 0.0
            break
        if list(got) != _oracle(types):
            ok = 0.0
            break

    return {"exact_match": ok}
