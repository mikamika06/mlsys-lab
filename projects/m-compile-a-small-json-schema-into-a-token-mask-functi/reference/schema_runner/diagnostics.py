"""Diagnostics for unsatisfiable or non-terminating schema constraints."""

from typing import Any, Dict, List, Tuple
from schema_runner.compiler import SchemaMaskCompiler


def diagnose_schema_deadlock(
    vocab: Dict[int, str],
    eos_token_id: int,
    schema: Dict[str, Any],
    max_depth: int = 100,
) -> Tuple[bool, str]:
    if schema.get("type") == "object" and "properties" in schema:
        props = schema.get("properties", {})
        req = schema.get("required", [])
        for r in req:
            if r not in props:
                return True, f"Required property '{r}' not defined in properties"
        if schema.get("additionalProperties") is False and len(props) == 0 and len(req) > 0:
            return True, "Unsatisfiable schema: required properties on empty object"

    compiler = SchemaMaskCompiler(vocab, eos_token_id)
    mask_fn = compiler.compile(schema)

    current_tokens = []
    visited = set()

    for step in range(max_depth):
        allowed = mask_fn(current_tokens)
        if not allowed:
            return True, f"Deadlock reached at step {step}: no allowed continuation tokens"
        if eos_token_id in allowed:
            return False, "Schema is satisfiable and can reach EOS"

        next_tok = sorted(allowed)[0]
        state_key = (tuple(current_tokens), next_tok)
        if state_key in visited:
            return True, "Infinite loop detected without reaching EOS"
        visited.add(state_key)
        current_tokens.append(next_tok)

    return True, f"Exceeded max depth {max_depth} without reaching EOS"
