import dis


def _oracle(func):
    return [ins.opname for ins in dis.get_instructions(func)]


def _fixture(value):
    total = 0
    for i in range(value):
        if i % 2 == 0:
            total += i
        else:
            total -= i
    return total


def grade(sol, fx) -> dict:
    expected = _oracle(_fixture)
    try:
        got = sol.opcode_sequence(_fixture)
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
