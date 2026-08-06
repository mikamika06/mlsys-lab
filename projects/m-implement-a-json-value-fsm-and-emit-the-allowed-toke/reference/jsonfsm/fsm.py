import json

class JSONFSM:
    def __init__(self, schema):
        self.schema = schema
        self.state = "START"
        self.depth = 0

    def step(self, token_str):
        if self.state == "START":
            if token_str.strip() == "{":
                self.state = "OBJECT_KEY"
            elif token_str.strip() == "[":
                self.state = "ARRAY_VAL"
        elif self.state == "OBJECT_KEY":
            if token_str.strip() == "}":
                self.state = "DONE"
            elif token_str.startswith('"'):
                self.state = "OBJECT_COLON"
        elif self.state == "OBJECT_COLON":
            if token_str.strip() == ":":
                self.state = "OBJECT_VAL"
        elif self.state == "OBJECT_VAL":
            if token_str.strip() in ('"', "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "true", "false", "null"):
                self.state = "OBJECT_COMMA"
        elif self.state == "OBJECT_COMMA":
            if token_str.strip() == ",":
                self.state = "OBJECT_KEY"
            elif token_str.strip() == "}":
                self.state = "DONE"
        return self.state

    def allowed_tokens(self, vocab_tokens):
        allowed = []
        for idx, tok in enumerate(vocab_tokens):
            if self.state == "START" and tok.strip() in ("{", "["):
                allowed.append(idx)
            elif self.state == "OBJECT_KEY" and (tok.strip().startswith('"') or tok.strip() == "}"):
                allowed.append(idx)
            elif self.state == "OBJECT_COLON" and tok.strip() == ":":
                allowed.append(idx)
            elif self.state == "OBJECT_VAL" and tok.strip() in ('"', "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "true", "false", "null"):
                allowed.append(idx)
            elif self.state == "OBJECT_COMMA" and tok.strip() in (",", "}"):
                allowed.append(idx)
            elif self.state == "DONE":
                pass
        if not allowed:
            allowed = list(range(len(vocab_tokens)))
        return allowed
