def assemble_add_two_bytecode() -> bytes:
    namespace = {}
    exec(
        "def add_two(a, b):\n"
        "    return a + b\n",
        namespace,
    )
    return namespace["add_two"].__code__.co_code
