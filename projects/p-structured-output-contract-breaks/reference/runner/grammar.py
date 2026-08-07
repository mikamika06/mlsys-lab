import json
import numpy as np

class JSONGrammarMasker:
    def __init__(self, schema: dict, vocab: list):
        self.schema = schema
        self.vocab = vocab
        self.req_fields = schema.get("required", [])
        self.properties = schema.get("properties", {})

    def get_allowed_token_ids(self, current_prefix: str) -> list:
        allowed = []
        for idx, token in enumerate(self.vocab):
            candidate = current_prefix + token
            if self._is_valid_prefix(candidate):
                allowed.append(idx)
        return allowed

    def _is_valid_prefix(self, candidate: str) -> bool:
        s = candidate.lstrip()
        if not s:
            return True
        if not s.startswith("{"):
            return False

        if s == "{":
            return True

        body = s[1:]
        if ":" not in body:
            key_part = body.strip()
            if key_part.startswith('"'):
                return True
            return False

        parts = body.split(":", 1)
        key = parts[0].strip().strip('"')
        val_part = parts[1].strip()

        if key not in self.properties:
            return False

        expected_type = self.properties[key].get("type")
        if not val_part:
            return True

        if expected_type == "integer":
            if val_part.startswith("-"):
                test_val = val_part[1:]
            else:
                test_val = val_part
            if test_val == "":
                return True
            if test_val.endswith("}"):
                test_val = test_val[:-1].strip()
            return test_val.isdigit()

        if expected_type == "string":
            if not val_part.startswith('"'):
                return False
            if val_part.count('"') == 1:
                return True
            if val_part.count('"') == 2 and val_part.endswith('"'):
                return True
            if val_part.count('"') == 2 and val_part.endswith("}"):
                return True
            return False

        return True

    def apply_mask(self, logits, current_prefix: str):
        allowed_ids = self.get_allowed_token_ids(current_prefix)
        masked_logits = np.full_like(logits, -1e9)
        if allowed_ids:
            masked_logits[allowed_ids] = logits[allowed_ids]
        return masked_logits
