import json
from typing import Any, Dict, List, Tuple

TOOL_SCHEMAS = {
    "get_weather": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer"},
        },
        "required": ["location"],
        "additionalProperties": False,
    },
    "create_user": {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "age": {"type": "integer"},
            "roles": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["username", "age"],
        "additionalProperties": False,
    },
}

TEST_TOOL_CALLS = [
    {
        "name": "get_weather",
        "arguments": '{"location": "Seattle", "unit": "celsius"}',
        "expected_valid": True,
    },
    {
        "name": "get_weather",
        "arguments": '{"location": "Seattle", "unit": "kelvin"}',
        "expected_valid": False,
    },
    {
        "name": "get_weather",
        "arguments": '{"unit": "celsius"}',
        "expected_valid": False,
    },
    {
        "name": "create_user",
        "arguments": '{"username": "alice", "age": 30, "roles": ["admin", "user"]}',
        "expected_valid": True,
    },
    {
        "name": "create_user",
        "arguments": '{"username": "bob", "age": "twenty"}',
        "expected_valid": False,
    },
    {
        "name": "non_existent_tool",
        "arguments": "{}",
        "expected_valid": False,
    },
]

RAW_RESPONSES = [
    {
        "id": "resp_1",
        "format_json": True,
        "content": '{"username": "charlie", "age": "thirty"}',
    },
    {
        "id": "resp_2",
        "format_json": True,
        "content": '{"username": "dave"}',
    },
    {
        "id": "resp_3",
        "format_json": True,
        "content": '{"username": "eve", "age": 25, "roles": ["user"]}',
    },
    {
        "id": "resp_4",
        "format_json": True,
        "content": "not a json string",
    },
]

USER_SCHEMA = TOOL_SCHEMAS["create_user"]


def reference_validate_value(val: Any, schema: Dict[str, Any]) -> List[str]:
    errors = []
    stype = schema.get("type")

    if stype == "object":
        if not isinstance(val, dict):
            return [f"expected object, got {type(val).__name__}"]
        req = schema.get("required", [])
        for r in req:
            if r not in val:
                errors.append(f"missing required property: {r}")
        props = schema.get("properties", {})
        add_props = schema.get("additionalProperties", True)
        for k, v in val.items():
            if k in props:
                errors.extend(reference_validate_value(v, props[k]))
            elif add_props is False:
                errors.append(f"unexpected property: {k}")

    elif stype == "array":
        if not isinstance(val, list):
            return [f"expected array, got {type(val).__name__}"]
        if "items" in schema:
            for item in val:
                errors.extend(reference_validate_value(item, schema["items"]))

    elif stype == "string":
        if not isinstance(val, str):
            errors.append(f"expected string, got {type(val).__name__}")
        elif "enum" in schema and val not in schema["enum"]:
            errors.append(f"value {val} not in enum {schema['enum']}")

    elif stype == "integer":
        if not (isinstance(val, int) and not isinstance(val, bool)):
            errors.append(f"expected integer, got {type(val).__name__}")

    return errors


def reference_validate_tool_call(
    tool_call: Dict[str, Any], schemas: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    name = tool_call.get("name")
    if not name or name not in schemas:
        return False, [f"Unknown or missing tool name: {name}"]

    schema = schemas[name]
    raw_args = tool_call.get("arguments", {})

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except Exception as e:
            return False, [f"JSON decode error: {str(e)}"]
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return False, ["Invalid arguments format"]

    errs = reference_validate_value(args, schema)
    return (len(errs) == 0, errs)


def reference_demonstrate_format_json_non_conformance(
    raw_responses: List[Dict[str, Any]], schema: Dict[str, Any]
) -> List[Dict[str, Any]]:
    results = []
    for resp in raw_responses:
        content = resp.get("content", "")
        is_valid_json = False
        parsed_data = None
        json_err = None

        try:
            parsed_data = json.loads(content)
            is_valid_json = True
        except Exception as e:
            json_err = str(e)

        schema_errors = []
        is_schema_valid = False
        if is_valid_json and isinstance(parsed_data, dict):
            schema_errors = reference_validate_value(parsed_data, schema)
            is_schema_valid = len(schema_errors) == 0

        results.append(
            {
                "id": resp.get("id"),
                "format_json_active": resp.get("format_json", False),
                "is_valid_json": is_valid_json,
                "is_schema_valid": is_schema_valid,
                "json_error": json_err,
                "schema_errors": schema_errors,
                "parsed": parsed_data,
            }
        )
    return results


def reference_roundtrip_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    def to_spec(s):
        spec = {"kind": "constrained_grammar", "target_type": s.get("type", "object"), "rules": {}}
        if "properties" in s:
            spec["rules"]["properties"] = {k: to_spec(v) for k, v in s["properties"].items()}
        if "required" in s:
            spec["rules"]["required_fields"] = sorted(list(s["required"]))
        if "additionalProperties" in s:
            spec["rules"]["allow_additional"] = s["additionalProperties"]
        if "items" in s:
            spec["rules"]["items_spec"] = to_spec(s["items"])
        if "enum" in s:
            spec["rules"]["enum_choices"] = list(s["enum"])
        return spec

    def to_schema(spec):
        s = {"type": spec.get("target_type", "object")}
        r = spec.get("rules", {})
        if "properties" in r:
            s["properties"] = {k: to_schema(v) for k, v in r["properties"].items()}
        if "required_fields" in r:
            s["required"] = list(r["required_fields"])
        if "allow_additional" in r:
            s["additionalProperties"] = r["allow_additional"]
        if "items_spec" in r:
            s["items"] = to_schema(r["items_spec"])
        if "enum_choices" in r:
            s["enum"] = list(r["enum_choices"])
        return s

    return to_schema(to_spec(schema))
