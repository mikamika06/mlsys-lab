import sys

sys.path.insert(0, ".")
from cudagraphs.memory import fix_buffer_overwrites


def test_buffer_overwrite_remediation():
    operations = [
        {"id": 1, "op": "add", "inputs": ["x", "y"], "output": "buf1"},
        {"id": 2, "op": "relu", "inputs": ["buf1"], "output": "buf2"},
        {"id": 3, "op": "mul", "inputs": ["buf2", "z"], "output": "buf1"},
    ]
    unsafe_aliases = {"buf1"}

    fixed = fix_buffer_overwrites(operations, unsafe_aliases)

    for op in fixed:
        if op["id"] == 3:
            assert (
                op["output"] != "buf1"
            ), "Unsafe aliased buffer 'buf1' was not remmapped!"
            assert op.get("requires_copy") is True
