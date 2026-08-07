import json

class Validator:
    def __init__(self, schema):
        self.schema = schema

    def validate(self, data):
        for k, t in self.schema.items():
            if k not in data or not isinstance(data[k], t):
                return False
        return True

class RobustPipeline:
    def __init__(self, schema):
        self.schema = schema
        self.validator = Validator(schema)

    def run_batch(self, n):
        failures = 0
        for i in range(n):
            item = {"name": f"user_{i}", "age": i % 100}
            if not self.validator.validate(item):
                failures += 1
        return failures
