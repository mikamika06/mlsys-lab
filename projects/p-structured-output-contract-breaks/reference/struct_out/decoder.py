import json

class SchemaDecoder:
    def __init__(self, schema):
        self.schema = schema

    def decode(self, raw):
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return {}

    def validate(self, data):
        for k, t in self.schema.items():
            if k not in data or not isinstance(data[k], t):
                return False
        return True
