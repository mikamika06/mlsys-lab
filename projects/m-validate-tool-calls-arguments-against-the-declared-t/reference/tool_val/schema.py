from typing import Any, Dict


def schema_to_constrained_spec(schema: Dict[str, Any]) -> Dict[str, Any]:
    stype = schema.get("type", "object")
    spec = {
        "kind": "constrained_grammar",
        "target_type": stype,
        "rules": {},
    }

    if "properties" in schema:
        spec["rules"]["properties"] = {}
        for prop, p_schema in schema["properties"].items():
            spec["rules"]["properties"][prop] = schema_to_constrained_spec(
                p_schema
            )

    if "required" in schema:
        spec["rules"]["required_fields"] = sorted(list(schema["required"]))

    if "additionalProperties" in schema:
        spec["rules"]["allow_additional"] = schema["additionalProperties"]

    if "items" in schema:
        spec["rules"]["items_spec"] = schema_to_constrained_spec(schema["items"])

    if "enum" in schema:
        spec["rules"]["enum_choices"] = list(schema["enum"])

    return spec


def constrained_spec_to_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    stype = spec.get("target_type", "object")
    schema = {"type": stype}
    rules = spec.get("rules", {})

    if "properties" in rules:
        schema["properties"] = {}
        for prop, p_spec in rules["properties"].items():
            schema["properties"][prop] = constrained_spec_to_schema(p_spec)

    if "required_fields" in rules:
        schema["required"] = list(rules["required_fields"])

    if "allow_additional" in rules:
        schema["additionalProperties"] = rules["allow_additional"]

    if "items_spec" in rules:
        schema["items"] = constrained_spec_to_schema(rules["items_spec"])

    if "enum_choices" in rules:
        schema["enum"] = list(rules["enum_choices"])

    return schema


def roundtrip_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    spec = schema_to_constrained_spec(schema)
    return constrained_spec_to_schema(spec)
