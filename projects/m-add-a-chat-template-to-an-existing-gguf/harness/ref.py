class MockReaderField:
    def __init__(self, value_str: str):
        self.parts = [None, bytearray(value_str.encode("utf-8"))]

class MockReader:
    def __init__(self, fields_dict: dict):
        self.fields = {k: MockReaderField(v) for k, v in fields_dict.items()}

class MockWriter:
    def __init__(self):
        self.kv = {}
        
    def add_string(self, key, value):
        self.kv[key] = value
