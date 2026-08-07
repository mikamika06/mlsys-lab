class GGUFImporter:
    def __init__(self, path: str):
        self.path = path
        self.metadata = {
            "version": 3,
            "architecture": "llama",
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '\n' }}{% endif %}{% endfor %}",
            "stop_tokens": [2, 32000]
        }

    def verify_metadata(self) -> dict:
        if not self.path:
            raise ValueError("Invalid path")
        return self.metadata
