from gguf_chat.modifier import set_chat_template

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

def test_backup_preservation():
    reader = MockReader({
        "tokenizer.chat_template": "current_template",
        "tokenizer.chat_template.backup": "original_template"
    })
    writer = MockWriter()
    
    set_chat_template(reader, writer, "new_template")
    
    assert writer.kv.get("tokenizer.chat_template.backup") == "original_template"
    assert writer.kv.get("tokenizer.chat_template") == "new_template"
