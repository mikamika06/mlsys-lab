import json

class TruncationHandler:
    def __init__(self, schema):
        self.schema = schema

    def is_truncated(self, raw):
        try:
            json.loads(raw)
            return False
        except Exception:
            return "{" in raw and "}" not in raw

    def repair(self, raw):
        fixed = raw.strip()
        if not fixed.endswith("}"):
            fixed += "}"
        return fixed

    def validate(self, raw):
        try:
            data = json.loads(raw)
            for k, t in self.schema.items():
                if k not in data or not isinstance(data[k], t):
                    return False
            return True
        except Exception:
            return False
