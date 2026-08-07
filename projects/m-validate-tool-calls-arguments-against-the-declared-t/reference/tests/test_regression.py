import sys

sys.path.insert(0, ".")
from tool_val.validator import validate_tool_call


def test_validate_detects_type_mismatch():
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "tag": {"type": "string"},
        },
        "required": ["count"],
    }
    tc = {"name": "test_func", "arguments": '{"count": "10", "tag": "prod"}'}
    valid, errors = validate_tool_call(tc, {"test_func": schema})
    assert not valid, "Validation should fail for string count"
    assert len(errors) > 0


def test_validate_detects_missing_required():
    schema = {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    }
    tc = {"name": "test_func", "arguments": "{}"}
    valid, errors = validate_tool_call(tc, {"test_func": schema})
    assert not valid, "Validation should fail when required field is missing"
    assert len(errors) > 0


def test_validate_passes_on_correct_input():
    schema = {
        "type": "object",
        "properties": {"enabled": {"type": "boolean"}},
        "required": ["enabled"],
    }
    tc = {"name": "test_func", "arguments": '{"enabled": true}'}
    valid, errors = validate_tool_call(tc, {"test_func": schema})
    assert valid, f"Validation should pass but failed with {errors}"
    assert len(errors) == 0
