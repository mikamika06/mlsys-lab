"""Regression tests for schema compilation and deadlock diagnosis."""

from schema_runner.diagnostics import diagnose_schema_deadlock


def test_schema_deadlock_detection():
    vocab = {
        0: "{",
        1: '"',
        2: "a",
        3: '":',
        4: "0",
        5: "}",
        6: "<EOS>",
    }
    eos_id = 6

    unsatisfiable_schema = {
        "type": "object",
        "properties": {},
        "required": ["missing_field"],
    }

    is_deadlock, msg = diagnose_schema_deadlock(
        vocab, eos_id, unsatisfiable_schema
    )
    assert is_deadlock, "Failed to detect deadlock for unsatisfiable schema"
    assert len(msg) > 0
