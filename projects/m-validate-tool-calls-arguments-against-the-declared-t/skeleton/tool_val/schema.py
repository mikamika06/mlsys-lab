from typing import Any, Dict


def schema_to_constrained_spec(schema: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError


def constrained_spec_to_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError


def roundtrip_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError
