class GGUFTokenizer:
    def __init__(self, metadata: dict):
        raise NotImplementedError

    def apply_chat_template(self, messages: list) -> str:
        raise NotImplementedError

    def encode(self, text: str) -> list:
        raise NotImplementedError

    def get_stop_sequences(self) -> list:
        raise NotImplementedError
