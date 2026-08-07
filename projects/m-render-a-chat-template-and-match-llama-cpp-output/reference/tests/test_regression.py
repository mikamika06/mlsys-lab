import sys

sys.path.insert(0, ".")
from chattpl.patch import patch_messages


def test_patch_moves_system_messages():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late system"},
    ]
    patched = patch_messages(msgs)
    assert patched[0]["role"] == "system"
    assert patched[0]["content"] == "late system"
