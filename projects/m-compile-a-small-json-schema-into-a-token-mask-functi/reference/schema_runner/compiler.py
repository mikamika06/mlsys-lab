"""JSON Schema to token mask compiler."""

import json
from typing import Any, Callable, Dict, List, Set, Tuple


class SchemaMaskCompiler:
    def __init__(self, vocab: Dict[int, str], eos_token_id: int):
        self.vocab = vocab
        self.eos_token_id = eos_token_id

    def _get_valid_prefixes(self, schema: Dict[str, Any]) -> Set[str]:
        sType = schema.get("type", "object")
        valid = set()
        if sType == "object":
            props = schema.get("properties", {})
            req = schema.get("required", list(props.keys()))
            valid.add("")
            valid.add("{")
            items = []
            for k in sorted(props.keys()):
                p_type = props[k].get("type", "string")
                if p_type == "integer":
                    val = "0"
                elif p_type == "boolean":
                    val = "true"
                else:
                    val = '"a"'
                items.append(f'"{k}":{val}')
            full_json = "{" + ",".join(items) + "}"
            for i in range(1, len(full_json) + 1):
                valid.add(full_json[:i])
        elif sType == "string":
            valid.add("")
            valid.add('"')
            valid.add('"a"')
        return valid

    def compile(self, schema: Dict[str, Any]) -> Callable[[List[int]], Set[int]]:
        valid_prefixes = self._get_valid_prefixes(schema)

        def mask_fn(prefix_tokens: List[int]) -> Set[int]:
            text = "".join(self.vocab.get(t, "") for t in prefix_tokens)
            allowed = set()
            for tok_id, tok_str in self.vocab.items():
                cand = text + tok_str
                if cand in valid_prefixes:
                    allowed.add(tok_id)
            if text in valid_prefixes and any(text.endswith(c) for c in ["}", '"a"', "0", "true"]):
                allowed.add(self.eos_token_id)
            return allowed

        return mask_fn
