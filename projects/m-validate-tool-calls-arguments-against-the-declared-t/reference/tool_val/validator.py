import json
from typing import Any, Dict, List, Tuple


def _validate_value(
    val: Any, schema: Dict[str, Any], path: str = ""
) -> List[str]:
    errors = []
    stype = schema.get("type")

    if stype == "object":
        if not isinstance(val, dict):
            return [f"{path}: expected object, got {type(val).__name__}"]
        req = schema.get("required", [])
        for r in req:
            if r not in val:
                errors.append(
                    f"{path}.{r}" if path else f"{r}: missing required property"
                )
        props = schema.get("properties", {})
        add_props = schema.get("additionalProperties", True)
        for k, v in val.items():
            sub_path = f"{path}.{k}" if path else k
            if k in props:
                errors.extend(_validate_value(v, props[k], sub_path))
            elif add_props is False:
                errors.append(f"{sub_path}: unexpected property")
            elif isinstance(add_props, dict):
                errors.extend(_validate_value(v, add_props, sub_path))

    elif stype == "array":
        if not isinstance(val, list):
            return [f"{path}: expected array, got {type(val).__name__}"]
        if "items" in schema:
            for idx, item in enumerate(val):
                sub_path = f"{path}[{idx}]"
                errors.extend(_validate_value(item, schema["items"], sub_path))

    elif stype == "string":
        if not isinstance(val, str):
            errors.append(f"{path}: expected string, got {type(val).__name__}")
        elif "enum" in schema and val not in schema["enum"]:
            errors.append(f"{path}: value {val} not in enum {schema['enum']}")

    elif stype == "integer":
        if not (isinstance(val, int) and not isinstance(val, bool)):
            errors.append(f"{path}: expected integer, got {type(val).__name__}")

    elif stype == "number":
        if not (
            isinstance(val, (int, float)) and not isinstance(val, bool)
        ):
            errors.append(f"{path}: expected number, got {type(val).__name__}")

    elif stype == "boolean":
        if not isinstance(val, bool):
            errors.append(f"{path}: expected boolean, got {type(val).__name__}")

    return errors


def validate_tool_call(
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
            return False, [f"JSON decode error in arguments: {str(e)}"]
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return False, ["Arguments must be JSON string or dictionary"]

    errs = _validate_value(args, schema)
    return (len(errs) == 0, errs)


def demonstrate_format_json_non_conformance(
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
            schema_errors = _validate_value(parsed_data, schema)
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
