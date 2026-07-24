from mlsys import scorers


def _oracle() -> bytes:
    namespace = {}
    exec(
        "def add_two(a, b):\n"
        "    return a + b\n",
        namespace,
    )
    return namespace["add_two"].__code__.co_code


def grade(sol, fx) -> dict:
    try:
        got = sol.assemble_add_two_bytecode()
        if not isinstance(got, (bytes, bytearray)):
            return {"byte_exact_fraction": 0.0}
    except Exception:
        return {"byte_exact_fraction": 0.0}

    return {
        "byte_exact_fraction": scorers.byte_exact_fraction(
            _oracle(),
            bytes(got),
        )
    }
