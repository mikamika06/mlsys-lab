import dis


def _fixture_function(x):
    y = x * 2
    return y + 3


def _ref(fn):
    return [instruction.opname for instruction in dis.get_instructions(fn)]


def grade(sol, fx) -> dict:
    expected = _ref(_fixture_function)
    try:
        got = sol.disassemble_one_function(_fixture_function)
        ok = 1.0 if list(got) == expected else 0.0
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
