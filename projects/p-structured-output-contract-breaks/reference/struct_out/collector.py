import json

class FailureCollector:
    def __init__(self, schema):
        self.schema = schema

    def classify(self, raw):
        try:
            data = json.loads(raw)
            for k, t in self.schema.items():
                if k not in data or not isinstance(data[k], t):
                    return "type_mismatch"
            return "valid"
        except json.JSONDecodeError:
            if "{" in raw and "}" not in raw:
                return "truncated"
            return "extra_text"

def collect_and_classify(corpus, schema):
    fc = FailureCollector(schema)
    return {raw: fc.classify(raw) for raw, _ in corpus}
