class GGUFTokenizer:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.template = metadata.get("chat_template", "")

    def apply_chat_template(self, messages: list) -> str:
        res = ""
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                res += f"<|user|>\n{content}\n"
            elif role == "assistant":
                res += f"<|assistant|>\n{content}\n"
        return res

    def encode(self, text: str) -> list:
        return [ord(c) for c in text]

    def get_stop_sequences(self) -> list:
        return ["<|end|>", "<|user|>"]
